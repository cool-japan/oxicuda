//! Virtual memory management for fine-grained GPU address space control.
//!
//! This module provides abstractions for CUDA's virtual memory management
//! API (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, etc.), which
//! allows separating the concepts of virtual address reservation and
//! physical memory allocation.
//!
//! # Concepts
//!
//! * **Virtual Address Range** — A reservation of contiguous virtual
//!   addresses in the GPU address space. No physical memory is committed
//!   until explicitly mapped.
//!
//! * **Physical Allocation** — A chunk of physical GPU memory that can
//!   be mapped to one or more virtual address ranges.
//!
//! * **Mapping** — The association of a physical allocation with a region
//!   of a virtual address range.
//!
//! # Use Cases
//!
//! * **Sparse arrays** — Reserve a large virtual range but only commit
//!   physical memory for the tiles/pages that are actually used.
//!
//! * **Resizable buffers** — Reserve a large virtual range up-front and
//!   map additional physical memory as the buffer grows, without changing
//!   the base address.
//!
//! * **Multi-GPU memory** — Map physical allocations from different devices
//!   into the same virtual address space.
//!
//! # Status
//!
//! The virtual memory driver functions are not yet loaded in
//! `oxicuda-driver`. All operations that would require driver calls
//! currently return [`CudaError::NotSupported`]. The data structures
//! are fully functional for planning and validation purposes.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_memory::virtual_memory::{
//!     VirtualAddressRange, PhysicalAllocation, VirtualMemoryManager, AccessFlags,
//! };
//!
//! // Reserve 1 GiB of virtual address space with 2 MiB alignment.
//! let va = VirtualMemoryManager::reserve(1 << 30, 1 << 21)?;
//! assert_eq!(va.size(), 1 << 30);
//!
//! // The actual GPU calls are not yet available, so alloc/map/unmap
//! // return NotSupported.
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::fmt;

use oxicuda_driver::error::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// AccessFlags
// ---------------------------------------------------------------------------

/// Memory access permission flags for virtual memory mappings.
///
/// These flags control how a mapped virtual address range can be accessed
/// by a given device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AccessFlags {
    /// No access permitted. The mapping exists but cannot be read or written.
    #[default]
    None,
    /// Read-only access. The device can read but not write.
    Read,
    /// Full read-write access.
    ReadWrite,
}

impl fmt::Display for AccessFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Read => write!(f, "Read"),
            Self::ReadWrite => write!(f, "ReadWrite"),
        }
    }
}

// ---------------------------------------------------------------------------
// VirtualAddressRange
// ---------------------------------------------------------------------------

/// A reserved range of virtual addresses in the GPU address space.
///
/// This represents a contiguous block of virtual addresses that has been
/// reserved but not necessarily backed by physical memory. Physical memory
/// is associated with the range via [`VirtualMemoryManager::map`].
///
/// # Note
///
/// On systems without CUDA virtual memory support, the `base` address
/// is set to 0 and operations on the range will return
/// [`CudaError::NotSupported`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualAddressRange {
    base: u64,
    size: usize,
    alignment: usize,
}

impl VirtualAddressRange {
    /// Returns the base virtual address of the range.
    #[inline]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Returns the size of the range in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the alignment of the range in bytes.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns whether the range contains the given virtual address.
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base.saturating_add(self.size as u64)
    }

    /// Returns the end address (exclusive) of the range.
    #[inline]
    pub fn end(&self) -> u64 {
        self.base.saturating_add(self.size as u64)
    }
}

impl fmt::Display for VirtualAddressRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VA[0x{:016x}..0x{:016x}, {} bytes, align={}]",
            self.base,
            self.end(),
            self.size,
            self.alignment,
        )
    }
}

// ---------------------------------------------------------------------------
// PhysicalAllocation
// ---------------------------------------------------------------------------

/// A physical memory allocation on a specific GPU device.
///
/// Physical allocations represent actual GPU VRAM that can be mapped
/// into virtual address ranges. Multiple virtual ranges can map to
/// the same physical allocation (aliasing).
///
/// # Note
///
/// On systems without CUDA virtual memory support, the `handle` is
/// set to 0 and the allocation is not backed by real memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalAllocation {
    handle: u64,
    size: usize,
    device_ordinal: i32,
}

impl PhysicalAllocation {
    /// Returns the opaque handle for this physical allocation.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Returns the size of this allocation in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device ordinal this allocation belongs to.
    #[inline]
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }
}

impl fmt::Display for PhysicalAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PhysAlloc[handle=0x{:016x}, {} bytes, dev={}]",
            self.handle, self.size, self.device_ordinal,
        )
    }
}

// ---------------------------------------------------------------------------
// MappingRecord — tracks virtual-to-physical mappings
// ---------------------------------------------------------------------------

/// A record of a virtual-to-physical memory mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingRecord {
    /// Offset within the virtual address range where the mapping starts.
    pub va_offset: usize,
    /// Size of the mapped region in bytes.
    pub size: usize,
    /// Handle of the physical allocation backing this mapping.
    pub phys_handle: u64,
    /// Access permissions for this mapping.
    pub access: AccessFlags,
}

// ---------------------------------------------------------------------------
// VirtualMemoryManager
// ---------------------------------------------------------------------------

/// Manager for GPU virtual memory operations.
///
/// Provides methods for reserving virtual address ranges, allocating
/// physical memory, mapping/unmapping, and setting access permissions.
///
/// # Status
///
/// The underlying CUDA virtual memory driver functions
/// (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemUnmap`,
/// `cuMemSetAccess`) are not yet loaded in `oxicuda-driver`. All
/// methods that would require driver calls return
/// [`CudaError::NotSupported`].
///
/// The [`reserve`](Self::reserve) method creates a local placeholder
/// object for planning purposes.
pub struct VirtualMemoryManager;

impl VirtualMemoryManager {
    /// Reserves a range of virtual addresses in the GPU address space.
    ///
    /// The reserved range is not backed by physical memory until
    /// [`map`](Self::map) is called.
    ///
    /// # Parameters
    ///
    /// * `size` - Size of the virtual range to reserve in bytes.
    ///   Must be a multiple of `alignment`.
    /// * `alignment` - Alignment requirement in bytes. Must be a power
    ///   of two and non-zero.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero, `alignment` is
    ///   zero, `alignment` is not a power of two, or `size` is not a
    ///   multiple of `alignment`.
    pub fn reserve(size: usize, alignment: usize) -> CudaResult<VirtualAddressRange> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CudaError::InvalidValue);
        }
        if size % alignment != 0 {
            return Err(CudaError::InvalidValue);
        }

        // TODO: call cuMemAddressReserve when available in DriverApi.
        // For now, create a placeholder with a synthetic base address.
        // We use a deterministic address based on size+alignment for
        // reproducible testing.
        let synthetic_base = 0x0000_7F00_0000_0000_u64
            .wrapping_add(size as u64)
            .wrapping_add(alignment as u64);

        Ok(VirtualAddressRange {
            base: synthetic_base,
            size,
            alignment,
        })
    }

    /// Releases a previously reserved virtual address range.
    ///
    /// After this call, the virtual addresses are no longer reserved
    /// and may be reused by future reservations.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] because the driver function
    /// `cuMemAddressFree` is not yet loaded.
    pub fn release(_va: VirtualAddressRange) -> CudaResult<()> {
        // TODO: call cuMemAddressFree when available
        Err(CudaError::NotSupported)
    }

    /// Allocates physical memory on the specified device.
    ///
    /// The allocated memory is not accessible until mapped into a
    /// virtual address range via [`map`](Self::map).
    ///
    /// # Parameters
    ///
    /// * `size` - Size of the allocation in bytes. Must be non-zero.
    /// * `device_ordinal` - Ordinal of the device to allocate on.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    /// * [`CudaError::NotSupported`] because the driver function
    ///   `cuMemCreate` is not yet loaded.
    pub fn alloc_physical(size: usize, device_ordinal: i32) -> CudaResult<PhysicalAllocation> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }
        // TODO: call cuMemCreate when available in DriverApi
        Err(CudaError::NotSupported)?;

        // This code is unreachable today but shows the intended return type.
        Ok(PhysicalAllocation {
            handle: 0,
            size,
            device_ordinal,
        })
    }

    /// Frees a physical memory allocation.
    ///
    /// The allocation must not be currently mapped to any virtual range.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] because the driver function
    /// `cuMemRelease` is not yet loaded.
    pub fn free_physical(_phys: PhysicalAllocation) -> CudaResult<()> {
        // TODO: call cuMemRelease when available
        Err(CudaError::NotSupported)
    }

    /// Maps a physical allocation to a region of a virtual address range.
    ///
    /// After mapping, GPU kernels can access the virtual addresses and
    /// reads/writes will be routed to the physical memory.
    ///
    /// # Parameters
    ///
    /// * `va` - The virtual address range to map into.
    /// * `phys` - The physical allocation to map.
    /// * `offset` - Byte offset within the virtual range at which to
    ///   start the mapping. Must be aligned to the VA's alignment.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `offset` is not aligned, or if
    ///   the physical allocation would extend past the end of the virtual
    ///   range.
    /// * [`CudaError::NotSupported`] because the driver function
    ///   `cuMemMap` is not yet loaded.
    pub fn map(
        va: &VirtualAddressRange,
        phys: &PhysicalAllocation,
        offset: usize,
    ) -> CudaResult<()> {
        // Validate alignment
        if va.alignment > 0 && offset % va.alignment != 0 {
            return Err(CudaError::InvalidValue);
        }
        // Validate bounds
        let end = offset
            .checked_add(phys.size)
            .ok_or(CudaError::InvalidValue)?;
        if end > va.size {
            return Err(CudaError::InvalidValue);
        }
        // TODO: call cuMemMap when available
        Err(CudaError::NotSupported)
    }

    /// Unmaps a region of a virtual address range.
    ///
    /// After unmapping, accesses to the affected virtual addresses will
    /// fault. The physical memory is not freed — it can be remapped
    /// elsewhere.
    ///
    /// # Parameters
    ///
    /// * `va` - The virtual address range to unmap from.
    /// * `offset` - Byte offset within the range where unmapping starts.
    /// * `size` - Number of bytes to unmap.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the offset+size exceeds the
    ///   virtual range bounds.
    /// * [`CudaError::NotSupported`] because the driver function
    ///   `cuMemUnmap` is not yet loaded.
    pub fn unmap(va: &VirtualAddressRange, offset: usize, size: usize) -> CudaResult<()> {
        let end = offset.checked_add(size).ok_or(CudaError::InvalidValue)?;
        if end > va.size {
            return Err(CudaError::InvalidValue);
        }
        // TODO: call cuMemUnmap when available
        Err(CudaError::NotSupported)
    }

    /// Sets access permissions for a virtual address range on a device.
    ///
    /// This controls whether the specified device can read and/or write
    /// to the mapped virtual addresses.
    ///
    /// # Parameters
    ///
    /// * `va` - The virtual address range to set permissions on.
    /// * `device_ordinal` - The device to grant/deny access for.
    /// * `flags` - The access permission flags.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotSupported`] because the driver function
    /// `cuMemSetAccess` is not yet loaded.
    pub fn set_access(
        _va: &VirtualAddressRange,
        _device_ordinal: i32,
        _flags: AccessFlags,
    ) -> CudaResult<()> {
        // TODO: call cuMemSetAccess when available
        Err(CudaError::NotSupported)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_valid_range() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        let va = va.ok();
        assert!(va.is_some());
        if let Some(va) = va {
            assert_eq!(va.size(), 4096);
            assert_eq!(va.alignment(), 4096);
            assert!(va.base() > 0);
        }
    }

    #[test]
    fn reserve_zero_size_fails() {
        let result = VirtualMemoryManager::reserve(0, 4096);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_zero_alignment_fails() {
        let result = VirtualMemoryManager::reserve(4096, 0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_non_power_of_two_alignment_fails() {
        let result = VirtualMemoryManager::reserve(4096, 3);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_misaligned_size_fails() {
        // 4096+1 is not a multiple of 4096
        let result = VirtualMemoryManager::reserve(4097, 4096);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_large_range() {
        // Reserve 1 GiB with 2 MiB alignment.
        let gib = 1 << 30;
        let mib2 = 1 << 21;
        let va = VirtualMemoryManager::reserve(gib, mib2);
        assert!(va.is_ok());
        if let Ok(va) = va {
            assert_eq!(va.size(), gib);
            assert_eq!(va.alignment(), mib2);
        }
    }

    #[test]
    fn virtual_address_range_contains() {
        let va = VirtualMemoryManager::reserve(8192, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            assert!(va.contains(va.base()));
            assert!(va.contains(va.base() + 1));
            assert!(va.contains(va.base() + 8191));
            assert!(!va.contains(va.end()));
            assert!(!va.contains(va.base().wrapping_sub(1)));
        }
    }

    #[test]
    fn virtual_address_range_end() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            assert_eq!(va.end(), va.base() + 4096);
        }
    }

    #[test]
    fn virtual_address_range_display() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let disp = format!("{va}");
            assert!(disp.contains("VA["));
            assert!(disp.contains("4096 bytes"));
        }
    }

    #[test]
    fn alloc_physical_zero_size_fails() {
        let result = VirtualMemoryManager::alloc_physical(0, 0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn alloc_physical_returns_not_supported() {
        let result = VirtualMemoryManager::alloc_physical(4096, 0);
        assert_eq!(result, Err(CudaError::NotSupported));
    }

    #[test]
    fn release_returns_not_supported() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let result = VirtualMemoryManager::release(va);
            assert_eq!(result, Err(CudaError::NotSupported));
        }
    }

    #[test]
    fn map_validates_alignment() {
        let va = VirtualMemoryManager::reserve(8192, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let phys = PhysicalAllocation {
                handle: 1,
                size: 4096,
                device_ordinal: 0,
            };
            // Offset 1 is not aligned to 4096
            let result = VirtualMemoryManager::map(&va, &phys, 1);
            assert_eq!(result, Err(CudaError::InvalidValue));
        }
    }

    #[test]
    fn map_validates_bounds() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let phys = PhysicalAllocation {
                handle: 1,
                size: 8192, // larger than VA range
                device_ordinal: 0,
            };
            let result = VirtualMemoryManager::map(&va, &phys, 0);
            assert_eq!(result, Err(CudaError::InvalidValue));
        }
    }

    #[test]
    fn map_returns_not_supported_when_valid() {
        let va = VirtualMemoryManager::reserve(8192, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let phys = PhysicalAllocation {
                handle: 1,
                size: 4096,
                device_ordinal: 0,
            };
            let result = VirtualMemoryManager::map(&va, &phys, 0);
            assert_eq!(result, Err(CudaError::NotSupported));
        }
    }

    #[test]
    fn unmap_validates_bounds() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let result = VirtualMemoryManager::unmap(&va, 0, 8192);
            assert_eq!(result, Err(CudaError::InvalidValue));
        }
    }

    #[test]
    fn unmap_returns_not_supported_when_valid() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let result = VirtualMemoryManager::unmap(&va, 0, 4096);
            assert_eq!(result, Err(CudaError::NotSupported));
        }
    }

    #[test]
    fn set_access_returns_not_supported() {
        let va = VirtualMemoryManager::reserve(4096, 4096);
        assert!(va.is_ok());
        if let Ok(va) = va {
            let result = VirtualMemoryManager::set_access(&va, 0, AccessFlags::ReadWrite);
            assert_eq!(result, Err(CudaError::NotSupported));
        }
    }

    #[test]
    fn access_flags_default() {
        assert_eq!(AccessFlags::default(), AccessFlags::None);
    }

    #[test]
    fn access_flags_display() {
        assert_eq!(format!("{}", AccessFlags::None), "None");
        assert_eq!(format!("{}", AccessFlags::Read), "Read");
        assert_eq!(format!("{}", AccessFlags::ReadWrite), "ReadWrite");
    }

    #[test]
    fn physical_allocation_display() {
        let phys = PhysicalAllocation {
            handle: 0x1234,
            size: 4096,
            device_ordinal: 0,
        };
        let disp = format!("{phys}");
        assert!(disp.contains("4096 bytes"));
        assert!(disp.contains("dev=0"));
    }

    #[test]
    fn mapping_record_fields() {
        let record = MappingRecord {
            va_offset: 0,
            size: 4096,
            phys_handle: 42,
            access: AccessFlags::ReadWrite,
        };
        assert_eq!(record.va_offset, 0);
        assert_eq!(record.size, 4096);
        assert_eq!(record.phys_handle, 42);
        assert_eq!(record.access, AccessFlags::ReadWrite);
    }

    #[test]
    fn free_physical_returns_not_supported() {
        let phys = PhysicalAllocation {
            handle: 1,
            size: 4096,
            device_ordinal: 0,
        };
        let result = VirtualMemoryManager::free_physical(phys);
        assert_eq!(result, Err(CudaError::NotSupported));
    }
}
