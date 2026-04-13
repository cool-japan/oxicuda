//! Tiled matrix transpose kernel for multi-dimensional FFT.
//!
//! Generates a PTX kernel that performs an out-of-place matrix transpose
//! using 32x32 shared-memory tiles with +1 column padding to avoid
//! shared-memory bank conflicts.
//!
//! The transpose kernel is used between axis passes in 2-D and 3-D FFTs
//! to reorder data so that each 1-D FFT pass operates on contiguous memory.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::ptx_helpers::ptx_type_suffix;
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Tile dimension: each thread block transposes a 32x32 tile.
const TILE_DIM: u32 = 32;

/// Padded row stride in shared memory (avoids bank conflicts).
const TILE_STRIDE: u32 = TILE_DIM + 1;

/// Threads per block (32x8 launch configuration).
const BLOCK_DIM_X: u32 = TILE_DIM;
const BLOCK_DIM_Y: u32 = 8;

// ---------------------------------------------------------------------------
// Kernel generation
// ---------------------------------------------------------------------------

/// Generates a PTX kernel that transposes a `rows x cols` matrix of complex
/// elements.
///
/// Each complex element occupies two consecutive floats (re, im).  The
/// kernel operates on tiles of 32x32 complex elements, using shared memory
/// with +1 padding to eliminate bank conflicts.
///
/// # Kernel Parameters
///
/// | Name         | Type | Description                         |
/// |--------------|------|-------------------------------------|
/// | `input_ptr`  | u64  | Device pointer to the input matrix  |
/// | `output_ptr` | u64  | Device pointer to the output matrix |
/// | `rows`       | u32  | Number of rows in the input matrix  |
/// | `cols`       | u32  | Number of columns in the input      |
///
/// # Launch Configuration
///
/// - Grid: `(ceil(cols/32), ceil(rows/32), batch)`
/// - Block: `(32, 8, 1)`
///
/// # Errors
///
/// Returns [`PtxGenError`] if the PTX builder encounters an error.
pub fn generate_transpose_kernel(
    rows: u32,
    cols: u32,
    precision: FftPrecision,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("transpose_{suffix}_{rows}x{cols}");
    let elem_bytes = precision.element_bytes() as u32;

    // Shared memory for one tile: TILE_DIM * TILE_STRIDE complex elements
    // Each complex = 2 floats
    let shared_count = (TILE_DIM * TILE_STRIDE * 2) as usize;
    let float_ty = crate::ptx_helpers::ptx_float_type(precision);

    KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("rows", PtxType::U32)
        .param("cols", PtxType::U32)
        .shared_mem("tile", float_ty, shared_count)
        .max_threads_per_block(BLOCK_DIM_X * BLOCK_DIM_Y)
        .body(move |b| {
            b.comment(&format!(
                "Tiled transpose: {rows}x{cols}, tile={TILE_DIM}x{TILE_DIM}+1 padding"
            ));

            // Thread and block indices
            let tx = b.thread_id_x();
            let ty = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {ty}, %tid.y;"));
            let bx = b.block_id_x();
            let by = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {by}, %ctaid.y;"));

            // Load kernel parameters
            let input_ptr = b.load_param_u64("input_ptr");
            let output_ptr = b.load_param_u64("output_ptr");
            let in_rows = b.load_param_u32("rows");
            let in_cols = b.load_param_u32("cols");

            // Compute global column and base row for this block
            let global_col = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {global_col}, {bx}, {TILE_DIM}, {tx};"));

            let global_row_base = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {global_row_base}, {by}, {TILE_DIM}, {ty};"
            ));

            // Shared memory base
            let tile_base = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {tile_base}, tile;"));

            // Complex element byte size
            let complex_bytes = elem_bytes * 2;

            // Load tile: each thread loads TILE_DIM/BLOCK_DIM_Y = 4 rows
            let rows_per_thread = TILE_DIM / BLOCK_DIM_Y;
            for i in 0..rows_per_thread {
                let row = b.alloc_reg(PtxType::U32);
                let row_offset = i * BLOCK_DIM_Y;
                b.raw_ptx(&format!("add.u32 {row}, {global_row_base}, {row_offset};"));

                // Bounds check: row < rows && global_col < cols
                b.if_lt_u32(row.clone(), in_rows.clone(), |b| {
                    b.if_lt_u32(global_col.clone(), in_cols.clone(), |b| {
                        // Global linear index = row * cols + col
                        let linear_idx = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "mad.lo.u32 {linear_idx}, {row}, {in_cols}, {global_col};"
                        ));

                        // Byte offset in global memory
                        let cb = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {cb}, {complex_bytes};"));
                        let byte_off = b.mul_wide_u32_to_u64(linear_idx, cb);
                        let g_addr = b.add_u64(input_ptr.clone(), byte_off);

                        // Shared memory index: (ty + i*BLOCK_DIM_Y) * TILE_STRIDE + tx
                        // in units of complex elements, then * 2 for floats
                        let smem_idx = b.alloc_reg(PtxType::U32);
                        let local_row = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("add.u32 {local_row}, {ty}, {row_offset};"));
                        b.raw_ptx(&format!(
                            "mad.lo.u32 {smem_idx}, {local_row}, {}, {tx};",
                            TILE_STRIDE
                        ));

                        // Convert to byte offset in shared memory
                        let smem_byte_off = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "mul.lo.u32 {smem_byte_off}, {smem_idx}, {complex_bytes};"
                        ));
                        let smem_off_64 = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("cvt.u64.u32 {smem_off_64}, {smem_byte_off};"));
                        let s_addr = b.add_u64(tile_base.clone(), smem_off_64);

                        // Load real and imaginary parts
                        match precision {
                            FftPrecision::Single => {
                                let re = b.load_global_f32(g_addr.clone());
                                b.store_shared_f32(s_addr.clone(), re);
                                let im_g = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_g}, {g_addr}, {elem_bytes};"));
                                let im = b.load_global_f32(im_g);
                                let im_s = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_s}, {s_addr}, {elem_bytes};"));
                                b.store_shared_f32(im_s, im);
                            }
                            FftPrecision::Double => {
                                let re = b.load_global_f64(g_addr.clone());
                                let re_s = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("st.shared.f64 [{s_addr}], {re};"));
                                let _ = re_s;
                                let im_g = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_g}, {g_addr}, {elem_bytes};"));
                                let im = b.load_global_f64(im_g);
                                let im_s = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_s}, {s_addr}, {elem_bytes};"));
                                b.raw_ptx(&format!("st.shared.f64 [{im_s}], {im};"));
                            }
                        }
                    });
                });
            }

            // Synchronise: all threads have loaded their tile data
            b.bar_sync(0);

            // Store transposed tile: swap tx<->ty and bx<->by
            let out_col = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mad.lo.u32 {out_col}, {by}, {TILE_DIM}, {tx};"));

            let out_row_base = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {out_row_base}, {bx}, {TILE_DIM}, {ty};"
            ));

            for i in 0..rows_per_thread {
                let out_row = b.alloc_reg(PtxType::U32);
                let row_offset = i * BLOCK_DIM_Y;
                b.raw_ptx(&format!("add.u32 {out_row}, {out_row_base}, {row_offset};"));

                // Bounds: out_row < cols (transposed) && out_col < rows (transposed)
                b.if_lt_u32(out_row.clone(), in_cols.clone(), |b| {
                    b.if_lt_u32(out_col.clone(), in_rows.clone(), |b| {
                        // Read from shared memory (transposed): tile[tx][ty + i*BLOCK_DIM_Y]
                        let smem_idx = b.alloc_reg(PtxType::U32);
                        let local_col = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("add.u32 {local_col}, {ty}, {row_offset};"));
                        b.raw_ptx(&format!(
                            "mad.lo.u32 {smem_idx}, {tx}, {}, {local_col};",
                            TILE_STRIDE
                        ));

                        let smem_byte_off = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "mul.lo.u32 {smem_byte_off}, {smem_idx}, {complex_bytes};"
                        ));
                        let smem_off_64 = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("cvt.u64.u32 {smem_off_64}, {smem_byte_off};"));
                        let s_addr = b.add_u64(tile_base.clone(), smem_off_64);

                        // Global output index = out_row * rows + out_col
                        // (output is cols x rows)
                        let out_linear = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "mad.lo.u32 {out_linear}, {out_row}, {in_rows}, {out_col};"
                        ));

                        let cb2 = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {cb2}, {complex_bytes};"));
                        let out_byte_off = b.mul_wide_u32_to_u64(out_linear, cb2);
                        let g_out = b.add_u64(output_ptr.clone(), out_byte_off);

                        match precision {
                            FftPrecision::Single => {
                                let re = b.load_shared_f32(s_addr.clone());
                                b.store_global_f32(g_out.clone(), re);
                                let im_s = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_s}, {s_addr}, {elem_bytes};"));
                                let im = b.load_shared_f32(im_s);
                                let im_g = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_g}, {g_out}, {elem_bytes};"));
                                b.store_global_f32(im_g, im);
                            }
                            FftPrecision::Double => {
                                let re = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("ld.shared.f64 {re}, [{s_addr}];"));
                                b.raw_ptx(&format!("st.global.f64 [{g_out}], {re};"));
                                let im_s = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_s}, {s_addr}, {elem_bytes};"));
                                let im = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("ld.shared.f64 {im}, [{im_s}];"));
                                let im_g = b.alloc_reg(PtxType::U64);
                                b.raw_ptx(&format!("add.u64 {im_g}, {g_out}, {elem_bytes};"));
                                b.raw_ptx(&format!("st.global.f64 [{im_g}], {im};"));
                            }
                        }
                    });
                });
            }

            b.ret();
        })
        .build()
}

/// Returns the shared memory size in bytes for a transpose kernel.
#[allow(dead_code)]
pub fn transpose_shared_bytes(precision: FftPrecision) -> usize {
    // TILE_DIM * TILE_STRIDE complex elements * 2 (re+im) * element_bytes
    (TILE_DIM as usize) * (TILE_STRIDE as usize) * precision.complex_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_kernel_smoke() {
        let result = generate_transpose_kernel(64, 128, FftPrecision::Single, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("transpose_f32_64x128"));
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn transpose_kernel_f64() {
        let result = generate_transpose_kernel(256, 256, FftPrecision::Double, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("transpose_f64_256x256"));
        }
    }

    #[test]
    fn shared_bytes() {
        let bytes = transpose_shared_bytes(FftPrecision::Single);
        // 32 * 33 * 8 = 8448
        assert_eq!(bytes, 32 * 33 * 8);
    }
}
