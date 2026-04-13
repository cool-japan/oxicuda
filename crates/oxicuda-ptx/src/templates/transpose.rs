//! Coalesced shared-memory matrix transpose kernel template.
//!
//! This module generates a PTX kernel for transposing a 2D matrix using the
//! classic shared-memory tile technique. The kernel achieves coalesced global
//! memory reads and writes, and avoids shared-memory bank conflicts by padding
//! the tile dimension by one element (`TILE_DIM + 1`).
//!
//! # Algorithm
//!
//! 1. Each thread block loads a `TILE_DIM x TILE_DIM` tile from the input matrix
//!    into shared memory using coalesced reads (one row per warp).
//! 2. Each thread processes `TILE_DIM / BLOCK_ROWS` elements along the column
//!    direction, striding by `BLOCK_ROWS`.
//! 3. After a `bar.sync` barrier, threads write from shared memory to the output
//!    matrix in transposed order, again achieving coalesced writes.
//! 4. The shared memory tile is declared as `TILE_DIM * (TILE_DIM + 1)` to avoid
//!    bank conflicts when threads in a warp access the same column.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::transpose::TransposeTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = TransposeTemplate::new(PtxType::F32);
//! let ptx = template.generate(SmVersion::Sm80).expect("PTX generation failed");
//! assert!(ptx.contains("transpose_f32"));
//! assert!(ptx.contains(".shared"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Template for generating a coalesced shared-memory transpose kernel.
///
/// The kernel transposes a 2D matrix of the given precision using tiled shared
/// memory with bank-conflict-free padding. Each thread block handles one
/// `tile_dim x tile_dim` tile of the matrix.
#[derive(Debug, Clone)]
pub struct TransposeTemplate {
    /// The data precision for computation (e.g., `PtxType::F32` or `PtxType::F64`).
    pub precision: PtxType,
    /// Tile dimension — each thread block processes a `tile_dim x tile_dim` tile.
    /// Must be a power of 2 and >= 16. Default: 32.
    pub tile_dim: u32,
    /// Number of rows each thread block covers per iteration. Each thread handles
    /// `tile_dim / block_rows` elements. Must divide `tile_dim` evenly. Default: 8.
    pub block_rows: u32,
}

impl TransposeTemplate {
    /// Creates a new transpose template with default tile parameters.
    ///
    /// Defaults: `tile_dim = 32`, `block_rows = 8`.
    #[must_use]
    pub const fn new(precision: PtxType) -> Self {
        Self {
            precision,
            tile_dim: 32,
            block_rows: 8,
        }
    }

    /// Sets the tile dimension (each block handles `tile_dim x tile_dim` elements).
    #[must_use]
    pub const fn tile_dim(mut self, dim: u32) -> Self {
        self.tile_dim = dim;
        self
    }

    /// Sets the number of rows per thread block iteration.
    #[must_use]
    pub const fn block_rows(mut self, rows: u32) -> Self {
        self.block_rows = rows;
        self
    }

    /// Returns the kernel function name derived from the precision and tile config.
    ///
    /// The name follows the pattern `transpose_{type}` for default `tile_dim=32`,
    /// or `transpose_{type}_t{tile_dim}` for non-default tile dimensions.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        if self.tile_dim == 32 {
            format!("transpose_{type_str}")
        } else {
            format!("transpose_{type_str}_t{}", self.tile_dim)
        }
    }

    /// Validates the template configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - `precision` is not `F32` or `F64`
    /// - `tile_dim` is not a power of 2 or is less than 16
    /// - `block_rows` does not divide `tile_dim` evenly
    /// - `block_rows` is zero
    pub fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "transpose requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        if self.tile_dim < 16 {
            return Err(PtxGenError::GenerationFailed(format!(
                "tile_dim must be >= 16, got {}",
                self.tile_dim
            )));
        }
        if !self.tile_dim.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "tile_dim must be a power of 2, got {}",
                self.tile_dim
            )));
        }
        if self.block_rows == 0 {
            return Err(PtxGenError::GenerationFailed(
                "block_rows must be > 0".to_string(),
            ));
        }
        if self.tile_dim % self.block_rows != 0 {
            return Err(PtxGenError::GenerationFailed(format!(
                "block_rows ({}) must divide tile_dim ({}) evenly",
                self.block_rows, self.tile_dim
            )));
        }
        Ok(())
    }

    /// Generates the complete PTX module text for this transpose kernel.
    ///
    /// # Kernel parameters
    ///
    /// - `output`: pointer to output matrix (`.u64`)
    /// - `input`: pointer to input matrix (`.u64`)
    /// - `width`: number of columns in the input matrix (`.u32`)
    /// - `height`: number of rows in the input matrix (`.u32`)
    ///
    /// The output matrix has dimensions `height x width` (transposed).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] on validation failure or formatting error.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self, sm: SmVersion) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let tile_dim = self.tile_dim;
        let block_rows = self.block_rows;
        let iterations = tile_dim / block_rows;
        let kernel_name = self.kernel_name();

        // Shared memory: tile_dim * (tile_dim + 1) elements for bank-conflict-free access
        let smem_pitch = tile_dim + 1;
        let smem_elements = tile_dim * smem_pitch;
        let smem_bytes = smem_elements as usize * byte_size;

        // Register budget
        let reg_count: u32 = 32;
        let rd_count: u32 = 16;
        let val_count: u32 = (iterations + 4).max(8);
        let p_count: u32 = (iterations * 2 + 4).max(8);

        let mut ptx = String::with_capacity(4096);

        // Header
        emit_header(&mut ptx, sm)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_width,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_height").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Thread block dimensions
        writeln!(ptx, "    .maxntid {tile_dim}, {block_rows}, 1;")
            .map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .u32 %r<{reg_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .u64 %rd<{rd_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %val<{val_count}>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<{p_count}>;").map_err(PtxGenError::FormatError)?;

        // Shared memory with bank-conflict-free padding
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_tile[{}];",
            byte_size.max(4),
            smem_bytes
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load kernel parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r0, [%param_width];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r1, [%param_height];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute tile origin in the input matrix
        // block_x = ctaid.x * TILE_DIM
        // block_y = ctaid.y * TILE_DIM
        // tx = tid.x, ty = tid.y
        writeln!(ptx, "    // Compute tile origin and thread indices")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %tid.x;").map_err(PtxGenError::FormatError)?; // tx
        writeln!(ptx, "    mov.u32 %r3, %tid.y;").map_err(PtxGenError::FormatError)?; // ty
        writeln!(ptx, "    mov.u32 %r4, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r5, %ctaid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r6, %r4, {tile_dim};").map_err(PtxGenError::FormatError)?; // block_x
        writeln!(ptx, "    mul.lo.u32 %r7, %r5, {tile_dim};").map_err(PtxGenError::FormatError)?; // block_y
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Get shared memory base address
        writeln!(ptx, "    mov.u64 %rd2, smem_tile;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // === Phase 1: Load input tile into shared memory (coalesced reads) ===
        // For each iteration i in [0..iterations):
        //   input_col = block_x + tx
        //   input_row = block_y + ty + i * BLOCK_ROWS
        //   if (input_col < width && input_row < height):
        //     smem[(ty + i*BLOCK_ROWS) * (TILE_DIM+1) + tx] = input[input_row * width + input_col]
        writeln!(
            ptx,
            "    // Phase 1: Load tile from global to shared memory (coalesced)"
        )
        .map_err(PtxGenError::FormatError)?;

        // input_col = block_x + tx (constant across iterations)
        writeln!(ptx, "    add.u32 %r8, %r6, %r2;").map_err(PtxGenError::FormatError)?; // input_col

        for i in 0..iterations {
            let row_offset = i * block_rows;
            let pred_col = format!("%p{}", i * 2);
            let pred_row = format!("%p{}", i * 2 + 1);
            let val_reg = format!("%val{i}");

            writeln!(ptx).map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    // Iteration {i}: row offset = {row_offset}")
                .map_err(PtxGenError::FormatError)?;

            // input_row = block_y + ty + row_offset
            writeln!(ptx, "    add.u32 %r9, %r7, %r3;").map_err(PtxGenError::FormatError)?;
            if row_offset > 0 {
                writeln!(ptx, "    add.u32 %r9, %r9, {row_offset};")
                    .map_err(PtxGenError::FormatError)?;
            }

            // Bounds check: input_col < width && input_row < height
            writeln!(ptx, "    setp.lt.u32 {pred_col}, %r8, %r0;")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    setp.lt.u32 {pred_row}, %r9, %r1;")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    and.pred {pred_col}, {pred_col}, {pred_row};")
                .map_err(PtxGenError::FormatError)?;

            // Initialize value to zero (for out-of-bounds)
            emit_zero_mov(&mut ptx, &val_reg, self.precision)?;

            writeln!(ptx, "    @!{pred_col} bra $SKIP_LOAD_{i};")
                .map_err(PtxGenError::FormatError)?;

            // Global address: input + (input_row * width + input_col) * byte_size
            writeln!(ptx, "    mul.lo.u32 %r10, %r9, %r0;").map_err(PtxGenError::FormatError)?; // input_row * width
            writeln!(ptx, "    add.u32 %r10, %r10, %r8;").map_err(PtxGenError::FormatError)?; // + input_col
            writeln!(ptx, "    cvt.u64.u32 %rd3, %r10;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u64 %rd4, %rd1, %rd3;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.global{ty} {val_reg}, [%rd4];")
                .map_err(PtxGenError::FormatError)?;

            writeln!(ptx, "$SKIP_LOAD_{i}:").map_err(PtxGenError::FormatError)?;

            // Store to shared memory: smem[(ty + row_offset) * smem_pitch + tx]
            let smem_row = format!("ty_plus_{i}");
            let _ = smem_row; // naming reference only
            writeln!(ptx, "    add.u32 %r11, %r3, {row_offset};")
                .map_err(PtxGenError::FormatError)?; // ty + row_offset
            writeln!(ptx, "    mul.lo.u32 %r12, %r11, {smem_pitch};")
                .map_err(PtxGenError::FormatError)?; // * smem_pitch
            writeln!(ptx, "    add.u32 %r12, %r12, %r2;").map_err(PtxGenError::FormatError)?; // + tx
            writeln!(ptx, "    cvt.u64.u32 %rd5, %r12;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u64 %rd6, %rd2, %rd5;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd6], {val_reg};")
                .map_err(PtxGenError::FormatError)?;
        }

        writeln!(ptx).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    // Synchronize after loading tile").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // === Phase 2: Write from shared memory to output in transposed order ===
        // Output tile origin is at (block_y, block_x) in the output matrix (height x width -> width x height)
        // output_col = block_y + tx
        // output_row = block_x + ty + i * BLOCK_ROWS
        // smem read: smem[tx * smem_pitch + (ty + i*BLOCK_ROWS)]
        // (transposed read from shared memory)
        writeln!(
            ptx,
            "    // Phase 2: Write transposed tile from shared to global (coalesced)"
        )
        .map_err(PtxGenError::FormatError)?;

        // output_col = block_y + tx
        writeln!(ptx, "    add.u32 %r13, %r7, %r2;").map_err(PtxGenError::FormatError)?; // output_col

        for i in 0..iterations {
            let row_offset = i * block_rows;
            let pred_base = iterations * 2; // offset past phase 1 predicates
            let pred_col = format!("%p{}", pred_base + i * 2);
            let pred_row = format!("%p{}", pred_base + i * 2 + 1);
            let val_reg = format!("%val{i}");

            writeln!(ptx).map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    // Write iteration {i}: row offset = {row_offset}")
                .map_err(PtxGenError::FormatError)?;

            // output_row = block_x + ty + row_offset
            writeln!(ptx, "    add.u32 %r14, %r6, %r3;").map_err(PtxGenError::FormatError)?;
            if row_offset > 0 {
                writeln!(ptx, "    add.u32 %r14, %r14, {row_offset};")
                    .map_err(PtxGenError::FormatError)?;
            }

            // Bounds check: output_col < height && output_row < width
            // (output is width x height, so cols go up to height, rows up to width)
            writeln!(ptx, "    setp.lt.u32 {pred_col}, %r13, %r1;")
                .map_err(PtxGenError::FormatError)?; // output_col < height
            writeln!(ptx, "    setp.lt.u32 {pred_row}, %r14, %r0;")
                .map_err(PtxGenError::FormatError)?; // output_row < width
            writeln!(ptx, "    and.pred {pred_col}, {pred_col}, {pred_row};")
                .map_err(PtxGenError::FormatError)?;

            // Load from shared memory: smem[tx * smem_pitch + (ty + row_offset)]
            // This is the transposed read pattern
            writeln!(ptx, "    mul.lo.u32 %r15, %r2, {smem_pitch};")
                .map_err(PtxGenError::FormatError)?; // tx * smem_pitch
            writeln!(ptx, "    add.u32 %r16, %r3, {row_offset};")
                .map_err(PtxGenError::FormatError)?; // ty + row_offset
            writeln!(ptx, "    add.u32 %r15, %r15, %r16;").map_err(PtxGenError::FormatError)?; // tx * smem_pitch + ty + row_offset
            writeln!(ptx, "    cvt.u64.u32 %rd7, %r15;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {byte_size};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u64 %rd8, %rd2, %rd7;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} {val_reg}, [%rd8];")
                .map_err(PtxGenError::FormatError)?;

            // Store to output: output[output_row * height + output_col]
            // (output matrix is width x height, row-major)
            writeln!(ptx, "    @!{pred_col} bra $SKIP_STORE_{i};")
                .map_err(PtxGenError::FormatError)?;

            writeln!(ptx, "    mul.lo.u32 %r17, %r14, %r1;").map_err(PtxGenError::FormatError)?; // output_row * height
            writeln!(ptx, "    add.u32 %r17, %r17, %r13;").map_err(PtxGenError::FormatError)?; // + output_col
            writeln!(ptx, "    cvt.u64.u32 %rd9, %r17;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {byte_size};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add.u64 %rd10, %rd0, %rd9;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.global{ty} [%rd10], {val_reg};")
                .map_err(PtxGenError::FormatError)?;

            writeln!(ptx, "$SKIP_STORE_{i}:").map_err(PtxGenError::FormatError)?;
        }

        writeln!(ptx).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

/// Emits the PTX module header (version, target, address size).
fn emit_header(ptx: &mut String, sm: SmVersion) -> Result<(), PtxGenError> {
    writeln!(ptx, ".version {}", sm.ptx_version()).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits a `mov` instruction that loads zero into a floating-point register.
fn emit_zero_mov(ptx: &mut String, reg: &str, ty: PtxType) -> Result<(), PtxGenError> {
    match ty {
        PtxType::F32 => {
            writeln!(ptx, "    mov.f32 {reg}, 0f00000000;").map_err(PtxGenError::FormatError)?;
        }
        PtxType::F64 => {
            writeln!(ptx, "    mov.f64 {reg}, 0d0000000000000000;")
                .map_err(PtxGenError::FormatError)?;
        }
        _ => {
            return Err(PtxGenError::InvalidType(format!(
                "zero literal not supported for {}",
                ty.as_ptx_str()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let t = TransposeTemplate::new(PtxType::F32);
        assert_eq!(t.tile_dim, 32);
        assert_eq!(t.block_rows, 8);
        assert_eq!(t.precision, PtxType::F32);
    }

    #[test]
    fn test_builder_methods() {
        let t = TransposeTemplate::new(PtxType::F64)
            .tile_dim(16)
            .block_rows(4);
        assert_eq!(t.tile_dim, 16);
        assert_eq!(t.block_rows, 4);
        assert_eq!(t.precision, PtxType::F64);
    }

    #[test]
    fn test_kernel_name_default_tile() {
        let t = TransposeTemplate::new(PtxType::F32);
        assert_eq!(t.kernel_name(), "transpose_f32");
    }

    #[test]
    fn test_kernel_name_custom_tile() {
        let t = TransposeTemplate::new(PtxType::F64).tile_dim(16);
        assert_eq!(t.kernel_name(), "transpose_f64_t16");
    }

    #[test]
    fn test_kernel_name_f64() {
        let t = TransposeTemplate::new(PtxType::F64);
        assert_eq!(t.kernel_name(), "transpose_f64");
    }

    #[test]
    fn test_validate_ok_f32() {
        let t = TransposeTemplate::new(PtxType::F32);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_validate_ok_f64() {
        let t = TransposeTemplate::new(PtxType::F64);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_type() {
        let t = TransposeTemplate::new(PtxType::U32);
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("F32 or F64"), "unexpected error: {msg}");
    }

    #[test]
    fn test_validate_tile_dim_too_small() {
        let t = TransposeTemplate::new(PtxType::F32).tile_dim(8);
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("tile_dim must be >= 16"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_validate_tile_dim_not_power_of_two() {
        let t = TransposeTemplate::new(PtxType::F32).tile_dim(24);
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("power of 2"), "unexpected error: {msg}");
    }

    #[test]
    fn test_validate_block_rows_not_divisor() {
        let t = TransposeTemplate::new(PtxType::F32)
            .tile_dim(32)
            .block_rows(6);
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("divide"), "unexpected error: {msg}");
    }

    #[test]
    fn test_validate_block_rows_zero() {
        let t = TransposeTemplate::new(PtxType::F32).block_rows(0);
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("block_rows must be > 0"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_generate_f32_contains_shared_memory() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(ptx.contains(".shared"), "PTX must declare shared memory");
        assert!(ptx.contains("smem_tile"), "PTX must use smem_tile label");
    }

    #[test]
    fn test_generate_f32_bank_conflict_free_padding() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        // For tile_dim=32, F32 (4 bytes), shared memory should be 32 * 33 * 4 = 4224 bytes
        let expected_bytes = 32 * 33 * 4;
        let pattern = format!("smem_tile[{expected_bytes}]");
        assert!(
            ptx.contains(&pattern),
            "expected shared memory size {expected_bytes}, PTX:\n{ptx}"
        );
    }

    #[test]
    fn test_generate_f64_bank_conflict_free_padding() {
        let t = TransposeTemplate::new(PtxType::F64);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        // For tile_dim=32, F64 (8 bytes), shared memory should be 32 * 33 * 8 = 8448 bytes
        let expected_bytes = 32 * 33 * 8;
        let pattern = format!("smem_tile[{expected_bytes}]");
        assert!(
            ptx.contains(&pattern),
            "expected shared memory size {expected_bytes}, PTX:\n{ptx}"
        );
    }

    #[test]
    fn test_generate_f32_contains_kernel_name() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(
            ptx.contains("transpose_f32"),
            "kernel name not found in PTX"
        );
    }

    #[test]
    fn test_generate_f64_valid() {
        let t = TransposeTemplate::new(PtxType::F64);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(ptx.contains("transpose_f64"));
        assert!(ptx.contains(".f64"));
        assert!(ptx.contains("ld.global.f64"));
        assert!(ptx.contains("st.global.f64"));
    }

    #[test]
    fn test_generate_contains_bar_sync() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(
            ptx.contains("bar.sync"),
            "PTX must contain bar.sync for shared memory coherence"
        );
    }

    #[test]
    fn test_generate_contains_kernel_params() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(ptx.contains("%param_output"), "missing output param");
        assert!(ptx.contains("%param_input"), "missing input param");
        assert!(ptx.contains("%param_width"), "missing width param");
        assert!(ptx.contains("%param_height"), "missing height param");
    }

    #[test]
    fn test_generate_coalesced_reads_and_writes() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        // Must have both global loads and stores
        assert!(
            ptx.contains("ld.global.f32"),
            "missing coalesced global loads"
        );
        assert!(
            ptx.contains("st.global.f32"),
            "missing coalesced global stores"
        );
        // Must have shared memory loads and stores
        assert!(ptx.contains("ld.shared.f32"), "missing shared memory loads");
        assert!(
            ptx.contains("st.shared.f32"),
            "missing shared memory stores"
        );
    }

    #[test]
    fn test_generate_custom_tile_16() {
        let t = TransposeTemplate::new(PtxType::F32)
            .tile_dim(16)
            .block_rows(4);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(ptx.contains("transpose_f32_t16"));
        // 16 * 17 * 4 = 1088 bytes shared memory
        assert!(ptx.contains("smem_tile[1088]"));
    }

    #[test]
    fn test_generate_custom_tile_64() {
        let t = TransposeTemplate::new(PtxType::F32)
            .tile_dim(64)
            .block_rows(16);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(ptx.contains("transpose_f32_t64"));
        // 64 * 65 * 4 = 16640 bytes
        assert!(ptx.contains("smem_tile[16640]"));
    }

    #[test]
    fn test_generate_iterations_count() {
        // tile_dim=32, block_rows=8 -> 4 iterations
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        // Should have load iteration labels 0..3
        assert!(ptx.contains("$SKIP_LOAD_0:"));
        assert!(ptx.contains("$SKIP_LOAD_3:"));
        // Should have store iteration labels 0..3
        assert!(ptx.contains("$SKIP_STORE_0:"));
        assert!(ptx.contains("$SKIP_STORE_3:"));
    }

    #[test]
    fn test_generate_invalid_config_returns_error() {
        let t = TransposeTemplate::new(PtxType::U32);
        assert!(t.generate(SmVersion::Sm80).is_err());
    }

    #[test]
    fn test_generate_sm75() {
        let t = TransposeTemplate::new(PtxType::F32);
        let ptx = t.generate(SmVersion::Sm75).expect("Sm75 should work");
        assert!(ptx.contains(".target"));
    }

    #[test]
    fn test_generate_maxntid_matches_block_config() {
        let t = TransposeTemplate::new(PtxType::F32)
            .tile_dim(16)
            .block_rows(4);
        let ptx = t
            .generate(SmVersion::Sm80)
            .expect("generation should succeed");
        assert!(
            ptx.contains(".maxntid 16, 4, 1"),
            "maxntid should match tile_dim x block_rows"
        );
    }
}
