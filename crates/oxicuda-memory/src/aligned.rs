//! Aligned GPU memory allocation for optimal access patterns.
//!
//! This module provides [`AlignedBuffer<T>`], a device memory buffer that
//! guarantees a specific alignment for the starting address.  Proper alignment
//! is critical for coalesced memory accesses on GPUs — misaligned loads and
//! stores can incur extra memory transactions, significantly hurting
//! throughput.
//!
//! # Alignment options
//!
//! | Variant          | Bytes   | Use case                                    |
//! |------------------|---------|---------------------------------------------|
//! | `Default`        | 256     | CUDA's natural allocation alignment          |
//! | `Align256`       | 256     | Explicit 256-byte alignment                  |
//! | `Align512`       | 512     | Optimal for many GPU texture/surface ops     |
//! | `Align1024`      | 1024    | Large-stride access patterns                 |
//! | `Align4096`      | 4096    | Page-aligned for unified/mapped memory       |
//! | `Custom(n)`      | n       | User-specified (must be a power of two)      |
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::aligned::{Alignment, AlignedBuffer};
//! let buf = AlignedBuffer::<f32>::alloc(1024, Alignment::Align512)?;
//! assert!(buf.is_aligned());
//! assert_eq!(buf.as_device_ptr() % 512, 0);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::marker::PhantomData;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::CUdeviceptr;
#[cfg(not(target_os = "macos"))]
use oxicuda_driver::loader::try_driver;

// ---------------------------------------------------------------------------
// Alignment enum
// ---------------------------------------------------------------------------

/// Specifies the byte alignment for a device memory allocation.
///
/// All variants represent alignments that are powers of two.  The `Custom`
/// variant is validated at allocation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alignment {
    /// CUDA's default allocation alignment (typically 256 bytes).
    Default,
    /// 256-byte alignment.
    Align256,
    /// 512-byte alignment.
    Align512,
    /// 1024-byte alignment.
    Align1024,
    /// 4096-byte (page) alignment.
    Align4096,
    /// User-specified alignment in bytes (must be a power of two).
    Custom(usize),
}

impl Alignment {
    /// Returns the alignment in bytes.
    ///
    /// For [`Default`](Alignment::Default), this returns 256 (the typical CUDA
    /// allocation alignment).
    #[inline]
    pub fn bytes(&self) -> usize {
        match self {
            Self::Default => 256,
            Self::Align256 => 256,
            Self::Align512 => 512,
            Self::Align1024 => 1024,
            Self::Align4096 => 4096,
            Self::Custom(n) => *n,
        }
    }

    /// Returns `true` if the alignment value is a power of two.
    ///
    /// This is always `true` for the named variants and may be `false` for
    /// [`Custom`](Alignment::Custom) with an invalid value.
    #[inline]
    pub fn is_power_of_two(&self) -> bool {
        let b = self.bytes();
        b > 0 && (b & (b - 1)) == 0
    }

    /// Returns `true` if the given device pointer satisfies this alignment.
    #[inline]
    pub fn is_aligned(&self, ptr: u64) -> bool {
        let b = self.bytes() as u64;
        if b == 0 {
            return false;
        }
        (ptr % b) == 0
    }
}

impl std::fmt::Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "Default(256)"),
            Self::Align256 => write!(f, "256"),
            Self::Align512 => write!(f, "512"),
            Self::Align1024 => write!(f, "1024"),
            Self::Align4096 => write!(f, "4096"),
            Self::Custom(n) => write!(f, "Custom({n})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Maximum alignment we allow (256 MiB).  Anything beyond this is almost
/// certainly a programming error.
const MAX_ALIGNMENT: usize = 256 * 1024 * 1024;

/// Validates that an [`Alignment`] is a power of two and within a reasonable
/// range.
///
/// # Errors
///
/// Returns [`CudaError::InvalidValue`] if:
/// - The alignment is zero.
/// - The alignment is not a power of two.
/// - The alignment exceeds 256 MiB.
pub fn validate_alignment(alignment: &Alignment) -> CudaResult<()> {
    let b = alignment.bytes();
    if b == 0 {
        return Err(CudaError::InvalidValue);
    }
    if !alignment.is_power_of_two() {
        return Err(CudaError::InvalidValue);
    }
    if b > MAX_ALIGNMENT {
        return Err(CudaError::InvalidValue);
    }
    Ok(())
}

/// Rounds `bytes` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two; otherwise the result is unspecified.
///
/// # Examples
///
/// ```
/// # use oxicuda_memory::aligned::round_up_to_alignment;
/// assert_eq!(round_up_to_alignment(100, 256), 256);
/// assert_eq!(round_up_to_alignment(256, 256), 256);
/// assert_eq!(round_up_to_alignment(257, 256), 512);
/// assert_eq!(round_up_to_alignment(0, 256), 0);
/// ```
#[inline]
pub fn round_up_to_alignment(bytes: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return bytes;
    }
    let mask = alignment - 1;
    (bytes + mask) & !mask
}

/// Recommends an optimal [`Alignment`] for a type based on its size.
///
/// The heuristic prefers alignments that enable coalesced memory accesses:
///
/// - Types of 16 bytes or more benefit from 512-byte alignment because a
///   32-thread warp issuing 16-byte loads touches exactly 512 bytes.
/// - Types of 8 bytes benefit from 256-byte alignment (warp touches 256 bytes).
/// - Smaller types use the CUDA default (256 bytes).
pub fn optimal_alignment_for_type<T>() -> Alignment {
    let size = std::mem::size_of::<T>();
    if size >= 16 {
        Alignment::Align512
    } else if size >= 8 {
        Alignment::Align256
    } else {
        Alignment::Default
    }
}

/// Computes the smallest alignment that ensures coalesced memory access for a
/// given `access_width` (in bytes) across a warp of `warp_size` threads.
///
/// The coalesced access pattern requires that `warp_size * access_width` bytes
/// are naturally aligned to the segment boundary used by the memory controller.
/// This function returns the smallest power-of-two alignment that is at least
/// `warp_size * access_width` bytes, capped at 4096 (page alignment).
///
/// # Examples
///
/// ```
/// # use oxicuda_memory::aligned::coalesce_alignment;
/// // 32 threads × 4 bytes = 128 → rounded up to 128
/// assert_eq!(coalesce_alignment(4, 32), 128);
/// // 32 threads × 16 bytes = 512
/// assert_eq!(coalesce_alignment(16, 32), 512);
/// ```
pub fn coalesce_alignment(access_width: usize, warp_size: u32) -> usize {
    let total = (warp_size as usize).saturating_mul(access_width);
    if total == 0 {
        return 1;
    }
    // Round up to the next power of two.
    let pot = total.next_power_of_two();
    // Cap at page alignment.
    pot.min(4096)
}

// ---------------------------------------------------------------------------
// AlignmentInfo
// ---------------------------------------------------------------------------

/// Information about the alignment of an existing device pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentInfo {
    /// The device pointer that was inspected.
    pub ptr: CUdeviceptr,
    /// The largest power-of-two alignment that the pointer satisfies.
    pub natural_alignment: usize,
    /// Whether the pointer is 256-byte aligned.
    pub is_256_aligned: bool,
    /// Whether the pointer is 512-byte aligned.
    pub is_512_aligned: bool,
    /// Whether the pointer is page-aligned (4096 bytes).
    pub is_page_aligned: bool,
}

/// Inspects a device pointer and reports its alignment characteristics.
///
/// For a null (zero) pointer the natural alignment is reported as `usize::MAX`
/// because zero is trivially aligned to every power of two.
pub fn check_alignment(ptr: CUdeviceptr) -> AlignmentInfo {
    let natural = if ptr == 0 {
        // Zero is aligned to every power of two; report maximum.
        usize::MAX
    } else {
        // The largest power-of-two factor is 2^(trailing_zeros).
        1_usize << (ptr.trailing_zeros().min(63))
    };
    AlignmentInfo {
        ptr,
        natural_alignment: natural,
        is_256_aligned: (ptr % 256) == 0,
        is_512_aligned: (ptr % 512) == 0,
        is_page_aligned: (ptr % 4096) == 0,
    }
}

// ---------------------------------------------------------------------------
// AlignedBuffer<T>
// ---------------------------------------------------------------------------

/// A device memory buffer whose starting address is guaranteed to meet the
/// requested [`Alignment`].
///
/// Internally this may over-allocate by up to `alignment - 1` extra bytes and
/// offset the user-visible pointer so that it lands on an aligned boundary.
/// The extra bytes (if any) are reported by [`wasted_bytes`](Self::wasted_bytes).
///
/// The buffer frees the *original* (unaligned) allocation on [`Drop`].
pub struct AlignedBuffer<T: Copy> {
    /// The aligned device pointer presented to the user.
    ptr: CUdeviceptr,
    /// Number of `T` elements.
    len: usize,
    /// Total bytes allocated (may be larger than `len * size_of::<T>()`).
    allocated_bytes: usize,
    /// The alignment that was requested.
    alignment: Alignment,
    /// Byte offset from the raw allocation base to `ptr`.
    offset: usize,
    /// The raw allocation base pointer (what we pass to `cuMemFree`).
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    raw_ptr: CUdeviceptr,
    /// Phantom marker for `T`.
    _phantom: PhantomData<T>,
}

// SAFETY: Same reasoning as `DeviceBuffer<T>` — the `u64` device pointer
// handle is managed by the thread-safe CUDA driver.
unsafe impl<T: Copy + Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedBuffer<T> {}

impl<T: Copy> AlignedBuffer<T> {
    /// Allocates an aligned device buffer capable of holding `n` elements of
    /// type `T`.
    ///
    /// The returned buffer's device pointer is guaranteed to be aligned to
    /// `alignment.bytes()`.  The allocation may be slightly larger than
    /// `n * size_of::<T>()` to accommodate the alignment offset.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero, alignment is invalid, or
    ///   the byte-size computation overflows.
    /// * [`CudaError::OutOfMemory`] if the GPU cannot satisfy the allocation.
    pub fn alloc(n: usize, alignment: Alignment) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::InvalidValue);
        }
        validate_alignment(&alignment)?;

        let elem_bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;

        let align_bytes = alignment.bytes();

        // Over-allocate by (alignment - 1) so we can always find an aligned
        // address within the allocation.
        let extra = align_bytes.saturating_sub(1);
        let total_bytes = elem_bytes
            .checked_add(extra)
            .ok_or(CudaError::InvalidValue)?;

        #[cfg(target_os = "macos")]
        let (raw_ptr, aligned_ptr, offset) = {
            // On macOS there is no CUDA driver.  Simulate with a synthetic
            // pointer that mimics typical driver behaviour (256-byte aligned
            // base).  Tests can exercise the alignment arithmetic.
            let base: CUdeviceptr = 0x0000_0001_0000_0100; // 256-byte aligned
            let aligned = round_up_to_alignment(base as usize, align_bytes) as CUdeviceptr;
            let off = (aligned - base) as usize;
            (base, aligned, off)
        };

        #[cfg(not(target_os = "macos"))]
        let (raw_ptr, aligned_ptr, offset) = {
            let api = try_driver()?;
            let mut base: CUdeviceptr = 0;
            let rc = unsafe { (api.cu_mem_alloc_v2)(&mut base, total_bytes) };
            oxicuda_driver::check(rc)?;
            let aligned = round_up_to_alignment(base as usize, align_bytes) as CUdeviceptr;
            let off = (aligned - base) as usize;
            (base, aligned, off)
        };

        Ok(Self {
            ptr: aligned_ptr,
            len: n,
            allocated_bytes: total_bytes,
            alignment,
            offset,
            raw_ptr,
            _phantom: PhantomData,
        })
    }

    /// Returns the aligned device pointer.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Returns the number of `T` elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    ///
    /// In practice this is always `false` because [`alloc`](Self::alloc)
    /// rejects zero-length allocations.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the alignment that was requested.
    #[inline]
    pub fn alignment(&self) -> &Alignment {
        &self.alignment
    }

    /// Returns the number of bytes wasted for alignment padding.
    ///
    /// This is the difference between the total allocation size and the
    /// minimum required (`len * size_of::<T>()`).
    #[inline]
    pub fn wasted_bytes(&self) -> usize {
        let needed = self.len * std::mem::size_of::<T>();
        self.allocated_bytes.saturating_sub(needed)
    }

    /// Returns `true` if the buffer's device pointer satisfies the requested
    /// alignment.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.alignment.is_aligned(self.ptr)
    }

    /// Returns the total number of bytes that were allocated (including
    /// alignment padding).
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Returns the byte offset from the raw allocation base to the aligned
    /// pointer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl<T: Copy> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        // Free the *raw* (unaligned) allocation, not the offset pointer.
        #[cfg(not(target_os = "macos"))]
        {
            if let Ok(api) = try_driver() {
                let rc = unsafe { (api.cu_mem_free_v2)(self.raw_ptr) };
                if rc != 0 {
                    tracing::warn!(
                        cuda_error = rc,
                        ptr = self.raw_ptr,
                        aligned_ptr = self.ptr,
                        len = self.len,
                        "cuMemFree_v2 failed during AlignedBuffer drop"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Alignment enum tests -----------------------------------------------

    #[test]
    fn alignment_bytes_named_variants() {
        assert_eq!(Alignment::Default.bytes(), 256);
        assert_eq!(Alignment::Align256.bytes(), 256);
        assert_eq!(Alignment::Align512.bytes(), 512);
        assert_eq!(Alignment::Align1024.bytes(), 1024);
        assert_eq!(Alignment::Align4096.bytes(), 4096);
    }

    #[test]
    fn alignment_bytes_custom() {
        assert_eq!(Alignment::Custom(64).bytes(), 64);
        assert_eq!(Alignment::Custom(2048).bytes(), 2048);
    }

    #[test]
    fn alignment_is_power_of_two() {
        assert!(Alignment::Default.is_power_of_two());
        assert!(Alignment::Align256.is_power_of_two());
        assert!(Alignment::Align512.is_power_of_two());
        assert!(Alignment::Align1024.is_power_of_two());
        assert!(Alignment::Align4096.is_power_of_two());
        assert!(Alignment::Custom(128).is_power_of_two());
        assert!(!Alignment::Custom(0).is_power_of_two());
        assert!(!Alignment::Custom(3).is_power_of_two());
        assert!(!Alignment::Custom(100).is_power_of_two());
    }

    #[test]
    fn alignment_is_aligned() {
        let a256 = Alignment::Align256;
        assert!(a256.is_aligned(0));
        assert!(a256.is_aligned(256));
        assert!(a256.is_aligned(512));
        assert!(!a256.is_aligned(1));
        assert!(!a256.is_aligned(128));
        assert!(!a256.is_aligned(255));

        let a512 = Alignment::Align512;
        assert!(a512.is_aligned(0));
        assert!(a512.is_aligned(512));
        assert!(!a512.is_aligned(256));
    }

    // -- round_up_to_alignment tests ----------------------------------------

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up_to_alignment(0, 256), 0);
        assert_eq!(round_up_to_alignment(1, 256), 256);
        assert_eq!(round_up_to_alignment(100, 256), 256);
        assert_eq!(round_up_to_alignment(256, 256), 256);
        assert_eq!(round_up_to_alignment(257, 256), 512);
        assert_eq!(round_up_to_alignment(511, 512), 512);
        assert_eq!(round_up_to_alignment(512, 512), 512);
        assert_eq!(round_up_to_alignment(513, 512), 1024);
    }

    #[test]
    fn round_up_zero_alignment() {
        // Zero alignment should not modify the value.
        assert_eq!(round_up_to_alignment(42, 0), 42);
    }

    // -- validate_alignment tests -------------------------------------------

    #[test]
    fn validate_named_variants_ok() {
        assert!(validate_alignment(&Alignment::Default).is_ok());
        assert!(validate_alignment(&Alignment::Align256).is_ok());
        assert!(validate_alignment(&Alignment::Align512).is_ok());
        assert!(validate_alignment(&Alignment::Align1024).is_ok());
        assert!(validate_alignment(&Alignment::Align4096).is_ok());
    }

    #[test]
    fn validate_custom_ok() {
        assert!(validate_alignment(&Alignment::Custom(64)).is_ok());
        assert!(validate_alignment(&Alignment::Custom(128)).is_ok());
        assert!(validate_alignment(&Alignment::Custom(8192)).is_ok());
    }

    #[test]
    fn validate_custom_bad() {
        // Zero
        assert!(validate_alignment(&Alignment::Custom(0)).is_err());
        // Not power of two
        assert!(validate_alignment(&Alignment::Custom(3)).is_err());
        assert!(validate_alignment(&Alignment::Custom(100)).is_err());
        // Too large (> 256 MiB)
        assert!(validate_alignment(&Alignment::Custom(512 * 1024 * 1024)).is_err());
    }

    // -- optimal_alignment_for_type tests -----------------------------------

    #[test]
    fn optimal_alignment_small_types() {
        // f32 = 4 bytes → Default
        assert_eq!(optimal_alignment_for_type::<f32>(), Alignment::Default);
        // u8 = 1 byte → Default
        assert_eq!(optimal_alignment_for_type::<u8>(), Alignment::Default);
    }

    #[test]
    fn optimal_alignment_medium_types() {
        // f64 = 8 bytes → Align256
        assert_eq!(optimal_alignment_for_type::<f64>(), Alignment::Align256);
        // u64 = 8 bytes → Align256
        assert_eq!(optimal_alignment_for_type::<u64>(), Alignment::Align256);
    }

    #[test]
    fn optimal_alignment_large_types() {
        // [f32; 4] = 16 bytes → Align512
        assert_eq!(
            optimal_alignment_for_type::<[f32; 4]>(),
            Alignment::Align512
        );
        // [f64; 4] = 32 bytes → Align512
        assert_eq!(
            optimal_alignment_for_type::<[f64; 4]>(),
            Alignment::Align512
        );
    }

    // -- coalesce_alignment tests -------------------------------------------

    #[test]
    fn coalesce_basic() {
        // 32 threads × 4 bytes = 128
        assert_eq!(coalesce_alignment(4, 32), 128);
        // 32 threads × 8 bytes = 256
        assert_eq!(coalesce_alignment(8, 32), 256);
        // 32 threads × 16 bytes = 512
        assert_eq!(coalesce_alignment(16, 32), 512);
        // 32 threads × 32 bytes = 1024
        assert_eq!(coalesce_alignment(32, 32), 1024);
    }

    #[test]
    fn coalesce_caps_at_page() {
        // 64 threads × 128 bytes = 8192 → capped at 4096
        assert_eq!(coalesce_alignment(128, 64), 4096);
    }

    #[test]
    fn coalesce_zero_inputs() {
        assert_eq!(coalesce_alignment(0, 32), 1);
        assert_eq!(coalesce_alignment(4, 0), 1);
        assert_eq!(coalesce_alignment(0, 0), 1);
    }

    // -- check_alignment tests ----------------------------------------------

    #[test]
    fn check_alignment_page_aligned() {
        let info = check_alignment(4096);
        assert!(info.is_256_aligned);
        assert!(info.is_512_aligned);
        assert!(info.is_page_aligned);
        assert!(info.natural_alignment >= 4096);
    }

    #[test]
    fn check_alignment_512_not_page() {
        let info = check_alignment(512);
        assert!(info.is_256_aligned);
        assert!(info.is_512_aligned);
        assert!(!info.is_page_aligned);
        assert_eq!(info.natural_alignment, 512);
    }

    #[test]
    fn check_alignment_odd_ptr() {
        let info = check_alignment(0x0001_0001);
        assert!(!info.is_256_aligned);
        assert!(!info.is_512_aligned);
        assert!(!info.is_page_aligned);
        assert_eq!(info.natural_alignment, 1);
    }

    #[test]
    fn check_alignment_null() {
        let info = check_alignment(0);
        assert_eq!(info.natural_alignment, usize::MAX);
        assert!(info.is_256_aligned);
        assert!(info.is_512_aligned);
        assert!(info.is_page_aligned);
    }

    // -- AlignedBuffer tests (macOS synthetic) ------------------------------

    #[cfg(target_os = "macos")]
    mod buffer_tests {
        use super::super::*;

        #[test]
        fn alloc_default_alignment() {
            let buf = AlignedBuffer::<f32>::alloc(128, Alignment::Default);
            assert!(buf.is_ok());
            let buf = buf.unwrap_or_else(|_| panic!("alloc failed"));
            assert_eq!(buf.len(), 128);
            assert!(!buf.is_empty());
            assert!(buf.is_aligned());
        }

        #[test]
        fn alloc_512_alignment() {
            let buf = AlignedBuffer::<f32>::alloc(256, Alignment::Align512);
            assert!(buf.is_ok());
            let buf = buf.unwrap_or_else(|_| panic!("alloc failed"));
            assert!(buf.is_aligned());
            assert_eq!(buf.as_device_ptr() % 512, 0);
        }

        #[test]
        fn alloc_4096_alignment() {
            let buf = AlignedBuffer::<f64>::alloc(64, Alignment::Align4096);
            assert!(buf.is_ok());
            let buf = buf.unwrap_or_else(|_| panic!("alloc failed"));
            assert!(buf.is_aligned());
            assert_eq!(buf.as_device_ptr() % 4096, 0);
        }

        #[test]
        fn alloc_zero_elements_fails() {
            let result = AlignedBuffer::<f32>::alloc(0, Alignment::Default);
            assert!(result.is_err());
        }

        #[test]
        fn alloc_invalid_alignment_fails() {
            let result = AlignedBuffer::<f32>::alloc(64, Alignment::Custom(3));
            assert!(result.is_err());
        }

        #[test]
        fn wasted_bytes_at_least_zero() {
            let buf = AlignedBuffer::<f32>::alloc(128, Alignment::Align512)
                .unwrap_or_else(|_| panic!("alloc failed"));
            // Wasted bytes = allocated_bytes - (128 * 4)
            // allocated_bytes = 128*4 + (512 - 1) = 1023
            // wasted = 1023 - 512 = 511
            assert!(buf.wasted_bytes() <= buf.alignment().bytes());
        }

        #[test]
        fn alignment_accessor() {
            let buf = AlignedBuffer::<u8>::alloc(64, Alignment::Align1024)
                .unwrap_or_else(|_| panic!("alloc failed"));
            assert_eq!(*buf.alignment(), Alignment::Align1024);
        }
    }

    // -- Display -----------------------------------------------------------

    #[test]
    fn alignment_display() {
        assert_eq!(format!("{}", Alignment::Default), "Default(256)");
        assert_eq!(format!("{}", Alignment::Align256), "256");
        assert_eq!(format!("{}", Alignment::Align512), "512");
        assert_eq!(format!("{}", Alignment::Custom(128)), "Custom(128)");
    }
}
