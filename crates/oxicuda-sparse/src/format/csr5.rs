//! CSR5 sparse matrix format.
//!
//! CSR5 is a tile-based variant of CSR designed for load-balanced SpMV on GPUs.
//! It divides the non-zero elements into fixed-size tiles (typically 32 elements
//! wide, matching the warp size), with each tile assigned to one warp.
//!
//! The key insight is that tiles may straddle row boundaries. The `tile_desc`
//! array encodes where rows start and end within each tile, enabling each warp
//! to know exactly which rows its elements contribute to.
//!
//! ## Structure
//!
//! A `Csr5Matrix<T>` contains:
//! - The original CSR arrays (`row_ptr`, `col_idx`, `values`)
//! - `tile_ptr[num_tiles+1]`: maps tile index to starting element index
//! - `tile_desc[num_tiles]`: bit-packed descriptor encoding row boundaries
//!   within each tile (which rows start/end in this tile)
//! - `calibrator[rows]`: workspace for cross-tile row contribution merging
//!
//! ## Reference
//!
//! W. Liu and B. Vinter, "CSR5: An Efficient Storage Format for
//! Cross-Platform Sparse Matrix-Vector Multiplication", ICS 2015.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SparseError, SparseResult};

/// Number of non-zero elements per tile (warp width).
pub const CSR5_TILE_WIDTH: u32 = 32;

/// Number of rows encoded per tile descriptor segment.
/// Each tile can contain contributions to multiple rows;
/// this is the maximum tracked per tile.
pub const CSR5_SIGMA: u32 = 32;

/// Tile descriptor: bit-packed row boundary information for one tile.
///
/// The descriptor encodes:
/// - `y_offset`: the starting row offset (local within the tile's first row)
/// - `seg_offset`: bitmask indicating which lanes in the warp start a new row
/// - `empty_offset`: number of empty (padding) elements at the end of the tile
///
/// This allows each warp to determine which rows its non-zero elements belong
/// to and how to reduce partial sums correctly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct TileDescriptor {
    /// Bitmask: bit `i` is set if lane `i` starts a new row segment.
    pub seg_mask: u32,
    /// Index of the first row that this tile contributes to.
    pub first_row: u32,
}

// SAFETY: TileDescriptor is repr(C) with only u32 fields, trivially bitwise
// copyable and has no interior pointers -- safe for GPU memcpy.
unsafe impl Send for TileDescriptor {}
unsafe impl Sync for TileDescriptor {}

/// A sparse matrix in CSR5 format, stored on GPU.
///
/// CSR5 augments the standard CSR representation with tile-based metadata
/// for load-balanced GPU SpMV. The non-zero elements are logically divided
/// into tiles of `CSR5_TILE_WIDTH` elements each. Each tile maps to one
/// warp during SpMV execution.
pub struct Csr5Matrix<T: GpuFloat> {
    /// Number of rows in the matrix.
    rows: u32,
    /// Number of columns in the matrix.
    cols: u32,
    /// Total number of non-zeros.
    nnz: u32,
    /// Number of tiles.
    num_tiles: u32,

    // -- Original CSR arrays --
    /// CSR row pointer array of length `rows + 1`.
    row_ptr: DeviceBuffer<i32>,
    /// CSR column indices of length `nnz`.
    col_idx: DeviceBuffer<i32>,
    /// CSR values of length `nnz`.
    values: DeviceBuffer<T>,

    // -- CSR5 tile metadata --
    /// Tile pointer: maps tile index to starting nnz index.
    /// Length `num_tiles + 1`.
    tile_ptr: DeviceBuffer<u32>,
    /// Tile descriptors: row boundary information per tile.
    /// Length `num_tiles`.
    tile_desc: DeviceBuffer<TileDescriptor>,
    /// Calibrator: workspace for cross-tile row contribution merging.
    /// Length `rows`. Used during the "calibrate" phase of CSR5 SpMV.
    calibrator: DeviceBuffer<T>,
}

impl<T: GpuFloat> Csr5Matrix<T> {
    /// Constructs a CSR5 matrix from host-side CSR arrays.
    ///
    /// This performs the CSR-to-CSR5 conversion on the host, computing
    /// tile pointers and descriptors, then uploads everything to GPU memory.
    ///
    /// # Arguments
    ///
    /// * `rows` -- Number of rows.
    /// * `cols` -- Number of columns.
    /// * `row_ptr` -- CSR row pointer (length `rows + 1`).
    /// * `col_idx` -- CSR column indices (length `nnz`).
    /// * `values` -- CSR values (length `nnz`).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if inputs are inconsistent.
    /// Returns [`SparseError::Cuda`] on GPU allocation/transfer failure.
    pub fn from_csr_host(
        rows: u32,
        cols: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[T],
    ) -> SparseResult<Self> {
        // Validate inputs
        if rows == 0 || cols == 0 {
            return Err(SparseError::InvalidFormat(
                "rows and cols must be non-zero".to_string(),
            ));
        }
        if row_ptr.len() != rows as usize + 1 {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr length ({}) must be rows + 1 ({})",
                row_ptr.len(),
                rows as usize + 1
            )));
        }
        if col_idx.len() != values.len() {
            return Err(SparseError::InvalidFormat(format!(
                "col_idx length ({}) must equal values length ({})",
                col_idx.len(),
                values.len()
            )));
        }

        let nnz = col_idx.len() as u32;
        if nnz == 0 {
            return Err(SparseError::ZeroNnz);
        }

        // Compute tile metadata
        let num_tiles = nnz.div_ceil(CSR5_TILE_WIDTH);

        // Build tile_ptr: tile i spans elements [i*TILE_WIDTH .. (i+1)*TILE_WIDTH]
        let mut h_tile_ptr = Vec::with_capacity(num_tiles as usize + 1);
        for t in 0..=num_tiles {
            h_tile_ptr.push((t * CSR5_TILE_WIDTH).min(nnz));
        }

        // Build tile descriptors
        let mut h_tile_desc = Vec::with_capacity(num_tiles as usize);
        for t in 0..num_tiles {
            let tile_start = h_tile_ptr[t as usize];
            let tile_end = h_tile_ptr[t as usize + 1];

            // Find the first row that this tile contributes to
            let first_row = find_row_for_element(row_ptr, tile_start);

            // Build segment mask: bit i is set if element (tile_start + i) is
            // the first element of a new row
            let mut seg_mask: u32 = 0;
            for lane in 0..CSR5_TILE_WIDTH {
                let elem_idx = tile_start + lane;
                if elem_idx >= tile_end {
                    break;
                }
                // Check if this element starts a new row
                let elem_row = find_row_for_element(row_ptr, elem_idx);
                if lane == 0 {
                    // First lane always starts the first row segment
                    // (seg_mask bit 0 is set implicitly by first_row)
                } else {
                    let prev_row = find_row_for_element(row_ptr, elem_idx - 1);
                    if elem_row != prev_row {
                        seg_mask |= 1 << lane;
                    }
                }
            }

            h_tile_desc.push(TileDescriptor {
                seg_mask,
                first_row,
            });
        }

        // Calibrator: zero-initialized workspace
        let h_calibrator = vec![T::gpu_zero(); rows as usize];

        // Upload to GPU
        let d_row_ptr = DeviceBuffer::from_host(row_ptr)?;
        let d_col_idx = DeviceBuffer::from_host(col_idx)?;
        let d_values = DeviceBuffer::from_host(values)?;
        let d_tile_ptr = DeviceBuffer::from_host(&h_tile_ptr)?;
        let d_tile_desc = DeviceBuffer::from_host(&h_tile_desc)?;
        let d_calibrator = DeviceBuffer::from_host(&h_calibrator)?;

        Ok(Self {
            rows,
            cols,
            nnz,
            num_tiles,
            row_ptr: d_row_ptr,
            col_idx: d_col_idx,
            values: d_values,
            tile_ptr: d_tile_ptr,
            tile_desc: d_tile_desc,
            calibrator: d_calibrator,
        })
    }

    /// Constructs a CSR5 matrix from an existing GPU-resident CSR matrix.
    ///
    /// Downloads CSR data to host, computes tile metadata, then re-uploads
    /// everything.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on GPU transfer/allocation failure.
    pub fn from_csr(csr: &super::CsrMatrix<T>) -> SparseResult<Self> {
        let (h_row_ptr, h_col_idx, h_values) = csr.to_host()?;
        Self::from_csr_host(csr.rows(), csr.cols(), &h_row_ptr, &h_col_idx, &h_values)
    }

    // -- Accessors --

    /// Returns the number of rows.
    #[inline]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Returns the total non-zero count.
    #[inline]
    pub fn nnz(&self) -> u32 {
        self.nnz
    }

    /// Returns the number of tiles.
    #[inline]
    pub fn num_tiles(&self) -> u32 {
        self.num_tiles
    }

    /// Returns a reference to the CSR row pointer device buffer.
    #[inline]
    pub fn row_ptr(&self) -> &DeviceBuffer<i32> {
        &self.row_ptr
    }

    /// Returns a reference to the CSR column index device buffer.
    #[inline]
    pub fn col_idx(&self) -> &DeviceBuffer<i32> {
        &self.col_idx
    }

    /// Returns a reference to the values device buffer.
    #[inline]
    pub fn values(&self) -> &DeviceBuffer<T> {
        &self.values
    }

    /// Returns a reference to the tile pointer device buffer.
    #[inline]
    pub fn tile_ptr(&self) -> &DeviceBuffer<u32> {
        &self.tile_ptr
    }

    /// Returns a reference to the tile descriptor device buffer.
    #[inline]
    pub fn tile_desc(&self) -> &DeviceBuffer<TileDescriptor> {
        &self.tile_desc
    }

    /// Returns a reference to the calibrator device buffer.
    #[inline]
    pub fn calibrator(&self) -> &DeviceBuffer<T> {
        &self.calibrator
    }

    /// Downloads tile metadata to host for inspection.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on transfer failure.
    pub fn tile_metadata_to_host(&self) -> SparseResult<(Vec<u32>, Vec<TileDescriptor>)> {
        let mut h_tile_ptr = vec![0u32; self.tile_ptr.len()];
        let mut h_tile_desc = vec![TileDescriptor::default(); self.tile_desc.len()];

        self.tile_ptr.copy_to_host(&mut h_tile_ptr)?;
        self.tile_desc.copy_to_host(&mut h_tile_desc)?;

        Ok((h_tile_ptr, h_tile_desc))
    }
}

/// Binary search to find which row a given non-zero element index belongs to.
///
/// Given `row_ptr[0..rows+1]` and an element index `elem_idx`, returns the
/// row `r` such that `row_ptr[r] <= elem_idx < row_ptr[r+1]`.
fn find_row_for_element(row_ptr: &[i32], elem_idx: u32) -> u32 {
    let elem = elem_idx as i32;
    let num_rows = row_ptr.len() - 1;

    // Binary search: find the largest r such that row_ptr[r] <= elem
    let mut lo: usize = 0;
    let mut hi: usize = num_rows;

    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if row_ptr[mid] <= elem {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    lo as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lightweight CPU-only CSR5 reference structure for testing tile metadata.
    /// The main `Csr5Matrix<T>` requires GPU (DeviceBuffer). This struct
    /// enables verifying tile_count, tile_width, and tile_ptr logic without GPU.
    struct Csr5CpuRef {
        #[allow(dead_code)]
        pub n_rows: usize,
        #[allow(dead_code)]
        pub nnz: usize,
        pub tile_width: usize,
        pub tile_count: usize,
        /// tile_ptr[i] = first row index covered by tile i.
        /// tile_ptr[tile_count] = n_rows.
        pub tile_ptr: Vec<u32>,
    }

    impl Csr5CpuRef {
        /// Build CPU reference CSR5 metadata from host CSR arrays.
        fn from_csr(n_rows: usize, row_ptr: &[i32], nnz: usize) -> Self {
            let tile_width = CSR5_TILE_WIDTH as usize;
            let tile_count = nnz.div_ceil(tile_width);

            // tile_ptr[i] = the row that contains the first element of tile i
            let mut tile_ptr = Vec::with_capacity(tile_count + 1);
            for tile in 0..tile_count {
                let first_elem = (tile * tile_width) as u32;
                tile_ptr.push(find_row_for_element(row_ptr, first_elem));
            }
            tile_ptr.push(n_rows as u32);

            Self {
                n_rows,
                nnz,
                tile_width,
                tile_count,
                tile_ptr,
            }
        }
    }

    #[test]
    fn csr5_tile_width() {
        assert_eq!(CSR5_TILE_WIDTH, 32);
    }

    #[test]
    fn find_row_for_element_basic() {
        // 3x3 identity matrix: row_ptr = [0, 1, 2, 3]
        let row_ptr = [0i32, 1, 2, 3];
        assert_eq!(find_row_for_element(&row_ptr, 0), 0);
        assert_eq!(find_row_for_element(&row_ptr, 1), 1);
        assert_eq!(find_row_for_element(&row_ptr, 2), 2);
    }

    #[test]
    fn find_row_for_element_varying_density() {
        // Row 0: 3 nnz, Row 1: 1 nnz, Row 2: 2 nnz
        // row_ptr = [0, 3, 4, 6]
        let row_ptr = [0i32, 3, 4, 6];
        assert_eq!(find_row_for_element(&row_ptr, 0), 0);
        assert_eq!(find_row_for_element(&row_ptr, 1), 0);
        assert_eq!(find_row_for_element(&row_ptr, 2), 0);
        assert_eq!(find_row_for_element(&row_ptr, 3), 1);
        assert_eq!(find_row_for_element(&row_ptr, 4), 2);
        assert_eq!(find_row_for_element(&row_ptr, 5), 2);
    }

    #[test]
    fn tile_descriptor_default() {
        let td = TileDescriptor::default();
        assert_eq!(td.seg_mask, 0);
        assert_eq!(td.first_row, 0);
    }

    #[test]
    fn csr5_tile_count_computation() {
        // 10 nnz with tile width 32 => 1 tile
        let num_tiles = 10_u32.div_ceil(CSR5_TILE_WIDTH);
        assert_eq!(num_tiles, 1);

        // 64 nnz => 2 tiles
        let num_tiles = 64_u32.div_ceil(CSR5_TILE_WIDTH);
        assert_eq!(num_tiles, 2);

        // 33 nnz => 2 tiles
        let num_tiles = 33_u32.div_ceil(CSR5_TILE_WIDTH);
        assert_eq!(num_tiles, 2);
    }

    #[test]
    fn csr5_sigma_value() {
        assert_eq!(CSR5_SIGMA, 32);
    }

    // --- New tests using Csr5CpuRef ---

    #[test]
    fn csr5_tile_count_32_nnz() {
        // Exactly 32 nnz => 1 tile (32 / 32 = 1)
        // Use a 32-row diagonal matrix (each row has exactly 1 nnz)
        let n_rows = 32usize;
        let row_ptr: Vec<i32> = (0..=32_i32).collect();
        let mat = Csr5CpuRef::from_csr(n_rows, &row_ptr, 32);
        assert_eq!(mat.tile_count, 1, "32 nnz should yield exactly 1 tile");
    }

    #[test]
    fn csr5_tile_count_33_nnz() {
        // 33 nnz => ceil(33/32) = 2 tiles
        // Use a 33-row diagonal matrix (each row has exactly 1 nnz)
        let n_rows = 33usize;
        let row_ptr: Vec<i32> = (0..=33_i32).collect();
        let mat = Csr5CpuRef::from_csr(n_rows, &row_ptr, 33);
        assert_eq!(mat.tile_count, 2, "33 nnz should yield 2 tiles");
    }

    #[test]
    fn csr5_tile_count_1024_nnz() {
        // 1024 nnz => 1024/32 = 32 tiles (exact)
        let n_rows = 1024usize;
        let row_ptr: Vec<i32> = (0..=1024_i32).collect();
        let mat = Csr5CpuRef::from_csr(n_rows, &row_ptr, 1024);
        assert_eq!(mat.tile_count, 32, "1024 nnz should yield 32 tiles");
    }

    #[test]
    fn csr5_tile_width_is_32() {
        let n_rows = 4usize;
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let mat = Csr5CpuRef::from_csr(n_rows, &row_ptr, 4);
        assert_eq!(
            mat.tile_width, 32,
            "tile_width must always be 32 (warp size)"
        );
    }

    #[test]
    fn csr5_tile_ptr_maps_row_correctly() {
        // 4x4 identity: row_ptr = [0,1,2,3,4], 4 nnz => 1 tile
        // tile_ptr[0] should be 0 (tile 0 starts at nnz 0, which belongs to row 0)
        let n_rows = 4usize;
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let mat = Csr5CpuRef::from_csr(n_rows, &row_ptr, 4);
        assert_eq!(mat.tile_count, 1);
        assert_eq!(
            mat.tile_ptr[0], 0,
            "tile 0 starts at row 0 for a 4x4 identity matrix"
        );
        assert_eq!(
            mat.tile_ptr[1], n_rows as u32,
            "tile_ptr[tile_count] must equal n_rows"
        );
    }
}
