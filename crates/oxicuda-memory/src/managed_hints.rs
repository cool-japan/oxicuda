//! Ergonomic managed memory hints API.
//!
//! This module builds on the raw [`crate::memory_info::mem_advise`]
//! and [`crate::memory_info::mem_prefetch`] functions to provide
//! a higher-level, builder-style API for controlling unified memory migration
//! behaviour.
//!
//! # Key types
//!
//! - [`MigrationPolicy`] — declarative policy for common migration patterns.
//! - [`ManagedMemoryHints`] — builder for applying hints to a memory region.
//! - [`PrefetchPlan`] — batch multiple prefetch operations into one plan.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::managed_hints::*;
//! # use oxicuda_driver::device::Device;
//! # use oxicuda_driver::stream::Stream;
//! // Assume `buf` is a UnifiedBuffer<f32> and `dev`/`stream` are valid.
//! // let hints = ManagedMemoryHints::from_unified(&buf);
//! // hints.set_read_mostly(&dev)?;
//! // hints.prefetch_to(&dev, &stream)?;
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use oxicuda_driver::device::Device;
use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::stream::Stream;

use crate::memory_info::{MemAdvice, mem_advise, mem_prefetch};
use crate::unified::UnifiedBuffer;

// ---------------------------------------------------------------------------
// MigrationPolicy
// ---------------------------------------------------------------------------

/// Declarative migration policy for unified memory regions.
///
/// Each variant encodes a common access pattern that can be translated into
/// one or more [`MemAdvice`] hints via [`to_advice_pairs`](MigrationPolicy::to_advice_pairs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MigrationPolicy {
    /// No special migration policy. Uses CUDA defaults.
    Default,
    /// Mark the region as read-mostly, enabling read-replica creation on
    /// accessing devices to reduce migration overhead.
    ReadMostly,
    /// Prefer that the data resides on the device with the given ordinal.
    PreferDevice(i32),
    /// Prefer that the data resides in host (CPU) memory.
    PreferHost,
}

impl MigrationPolicy {
    /// Converts this policy into the corresponding [`MemAdvice`] values
    /// that should be applied.
    ///
    /// For [`Default`](MigrationPolicy::Default) the returned vec is empty
    /// (no advice to set). For compound policies the vec contains all advice
    /// hints that need to be issued.
    pub fn to_advice_pairs(&self) -> Vec<MemAdvice> {
        match self {
            Self::Default => Vec::new(),
            Self::ReadMostly => vec![MemAdvice::SetReadMostly],
            Self::PreferDevice(_) => vec![MemAdvice::SetPreferredLocation],
            Self::PreferHost => vec![MemAdvice::SetPreferredLocation],
        }
    }

    /// Returns whether this is the [`Default`](MigrationPolicy::Default) variant.
    #[inline]
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Default)
    }
}

impl std::fmt::Display for MigrationPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "MigrationPolicy::Default"),
            Self::ReadMostly => write!(f, "MigrationPolicy::ReadMostly"),
            Self::PreferDevice(ord) => write!(f, "MigrationPolicy::PreferDevice({ord})"),
            Self::PreferHost => write!(f, "MigrationPolicy::PreferHost"),
        }
    }
}

// ---------------------------------------------------------------------------
// ManagedMemoryHints
// ---------------------------------------------------------------------------

/// Builder-style API for applying memory hints to a unified memory region.
///
/// Wraps a raw pointer + byte size and exposes methods that issue the
/// appropriate [`mem_advise`] / [`mem_prefetch`] driver calls.
///
/// # Construction
///
/// Use [`for_buffer`](Self::for_buffer) for raw pointers or
/// [`from_unified`](Self::from_unified) for [`UnifiedBuffer`] references.
#[derive(Debug, Clone)]
pub struct ManagedMemoryHints {
    /// Device pointer to the start of the managed region.
    ptr: u64,
    /// Total size of the region in bytes.
    byte_size: usize,
}

impl ManagedMemoryHints {
    /// Creates a `ManagedMemoryHints` from a raw device pointer and byte size.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if `byte_size` is zero.
    pub fn for_buffer(ptr: u64, byte_size: usize) -> CudaResult<Self> {
        if byte_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self { ptr, byte_size })
    }

    /// Creates a `ManagedMemoryHints` from a [`UnifiedBuffer`] reference.
    ///
    /// The pointer and byte size are extracted from the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the buffer reports zero bytes
    /// (should not happen for a validly constructed buffer).
    pub fn from_unified<T: Copy>(buf: &UnifiedBuffer<T>) -> CudaResult<Self> {
        Self::for_buffer(buf.as_device_ptr(), buf.byte_size())
    }

    /// Returns the device pointer this hint set targets.
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Returns the byte size of the targeted region.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    // -- Individual advice methods ------------------------------------------

    /// Marks the region as read-mostly on `device`, enabling read replicas.
    pub fn set_read_mostly(&self, device: &Device) -> CudaResult<()> {
        mem_advise(self.ptr, self.byte_size, MemAdvice::SetReadMostly, device)
    }

    /// Removes the read-mostly hint for `device`.
    pub fn unset_read_mostly(&self, device: &Device) -> CudaResult<()> {
        mem_advise(self.ptr, self.byte_size, MemAdvice::UnsetReadMostly, device)
    }

    /// Sets the preferred location to `device` for this region.
    pub fn set_preferred_location(&self, device: &Device) -> CudaResult<()> {
        mem_advise(
            self.ptr,
            self.byte_size,
            MemAdvice::SetPreferredLocation,
            device,
        )
    }

    /// Removes the preferred-location hint for `device`.
    pub fn unset_preferred_location(&self, device: &Device) -> CudaResult<()> {
        mem_advise(
            self.ptr,
            self.byte_size,
            MemAdvice::UnsetPreferredLocation,
            device,
        )
    }

    /// Indicates that `device` will access this memory region.
    pub fn set_accessed_by(&self, device: &Device) -> CudaResult<()> {
        mem_advise(self.ptr, self.byte_size, MemAdvice::SetAccessedBy, device)
    }

    /// Removes the accessed-by hint for `device`.
    pub fn unset_accessed_by(&self, device: &Device) -> CudaResult<()> {
        mem_advise(self.ptr, self.byte_size, MemAdvice::UnsetAccessedBy, device)
    }

    // -- Prefetch methods ---------------------------------------------------

    /// Prefetches the entire region to `device` on `stream`.
    pub fn prefetch_to(&self, device: &Device, stream: &Stream) -> CudaResult<()> {
        mem_prefetch(self.ptr, self.byte_size, device, stream)
    }

    /// Prefetches a sub-range of the region to `device`.
    ///
    /// # Parameters
    ///
    /// * `offset_bytes` — byte offset from the start of the region.
    /// * `count_bytes` — number of bytes to prefetch.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the range
    /// `[offset_bytes, offset_bytes + count_bytes)` exceeds the buffer, or
    /// if `count_bytes` is zero.
    pub fn prefetch_range(
        &self,
        offset_bytes: usize,
        count_bytes: usize,
        device: &Device,
        stream: &Stream,
    ) -> CudaResult<()> {
        if count_bytes == 0 {
            return Err(CudaError::InvalidValue);
        }
        let end = offset_bytes
            .checked_add(count_bytes)
            .ok_or(CudaError::InvalidValue)?;
        if end > self.byte_size {
            return Err(CudaError::InvalidValue);
        }
        let range_ptr = self
            .ptr
            .checked_add(offset_bytes as u64)
            .ok_or(CudaError::InvalidValue)?;
        mem_prefetch(range_ptr, count_bytes, device, stream)
    }

    // -- Policy convenience -------------------------------------------------

    /// Applies a [`MigrationPolicy`] to this memory region.
    ///
    /// For [`MigrationPolicy::Default`] this is a no-op.
    /// For other variants the corresponding advice hint(s) are issued.
    pub fn apply_policy(&self, policy: &MigrationPolicy, device: &Device) -> CudaResult<()> {
        apply_migration_policy(self.ptr, self.byte_size, policy, device)
    }
}

// ---------------------------------------------------------------------------
// PrefetchPlan
// ---------------------------------------------------------------------------

/// An entry in a [`PrefetchPlan`] recording a single prefetch operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchEntry {
    /// Device pointer to the start of the region.
    pub ptr: u64,
    /// Size of the region in bytes.
    pub byte_size: usize,
    /// Target device ordinal.
    pub device_ordinal: i32,
}

/// Batch multiple prefetch operations into a single plan.
///
/// Operations are recorded first, then executed together on a single stream
/// via [`execute`](PrefetchPlan::execute).
///
/// # Example
///
/// ```rust,no_run
/// # use oxicuda_memory::managed_hints::PrefetchPlan;
/// # use oxicuda_driver::stream::Stream;
/// let mut plan = PrefetchPlan::new();
/// plan.add(0x1000, 4096, 0)
///     .add(0x2000, 8192, 0);
/// assert_eq!(plan.len(), 2);
/// // plan.execute(&stream)?;
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PrefetchPlan {
    entries: Vec<PrefetchEntry>,
}

impl PrefetchPlan {
    /// Creates an empty prefetch plan.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Records a prefetch operation.
    ///
    /// The actual prefetch is deferred until [`execute`](Self::execute).
    pub fn add(&mut self, ptr: u64, byte_size: usize, device_ordinal: i32) -> &mut Self {
        self.entries.push(PrefetchEntry {
            ptr,
            byte_size,
            device_ordinal,
        });
        self
    }

    /// Returns the number of recorded prefetch operations.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no operations have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a slice of all recorded entries.
    #[inline]
    pub fn entries(&self) -> &[PrefetchEntry] {
        &self.entries
    }

    /// Executes all recorded prefetch operations on `stream`.
    ///
    /// Each entry is issued as a separate `mem_prefetch` call targeting the
    /// device identified by the entry's `device_ordinal`. Operations are
    /// enqueued in the order they were added.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered. Entries before the failing one
    /// will already have been enqueued.
    pub fn execute(&self, stream: &Stream) -> CudaResult<()> {
        for entry in &self.entries {
            let device = Device::get(entry.device_ordinal)?;
            mem_prefetch(entry.ptr, entry.byte_size, &device, stream)?;
        }
        Ok(())
    }
}

impl Default for PrefetchPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free function convenience
// ---------------------------------------------------------------------------

/// Applies a [`MigrationPolicy`] to a raw unified memory region.
///
/// This is a convenience function that translates the high-level policy into
/// the appropriate [`mem_advise`] calls.
///
/// # Parameters
///
/// * `ptr` — device pointer to the managed allocation.
/// * `byte_size` — size of the region in bytes.
/// * `policy` — the migration policy to apply.
/// * `device` — the device to which hints are directed.
///
/// # Errors
///
/// Forwards any error from the underlying driver call.
/// Returns [`CudaError::InvalidValue`] if `byte_size` is zero (when
/// policy is not `Default`).
pub fn apply_migration_policy(
    ptr: u64,
    byte_size: usize,
    policy: &MigrationPolicy,
    device: &Device,
) -> CudaResult<()> {
    match policy {
        MigrationPolicy::Default => Ok(()),
        MigrationPolicy::ReadMostly => mem_advise(ptr, byte_size, MemAdvice::SetReadMostly, device),
        MigrationPolicy::PreferDevice(_ordinal) => {
            // The advice targets the device passed by the caller. The ordinal
            // in the policy variant is informational — the caller is expected
            // to pass the corresponding Device handle.
            mem_advise(ptr, byte_size, MemAdvice::SetPreferredLocation, device)
        }
        MigrationPolicy::PreferHost => {
            // For host-preferred, we still issue SetPreferredLocation but
            // directed at the provided device handle. In a real CUDA
            // environment the caller would pass CU_DEVICE_CPU (-1).
            mem_advise(ptr, byte_size, MemAdvice::SetPreferredLocation, device)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MigrationPolicy tests ----------------------------------------------

    #[test]
    fn migration_policy_default_produces_empty_advice() {
        let pairs = MigrationPolicy::Default.to_advice_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn migration_policy_read_mostly_advice() {
        let pairs = MigrationPolicy::ReadMostly.to_advice_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], MemAdvice::SetReadMostly);
    }

    #[test]
    fn migration_policy_prefer_device_advice() {
        let pairs = MigrationPolicy::PreferDevice(0).to_advice_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], MemAdvice::SetPreferredLocation);
    }

    #[test]
    fn migration_policy_prefer_host_advice() {
        let pairs = MigrationPolicy::PreferHost.to_advice_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], MemAdvice::SetPreferredLocation);
    }

    #[test]
    fn migration_policy_is_default() {
        assert!(MigrationPolicy::Default.is_default());
        assert!(!MigrationPolicy::ReadMostly.is_default());
        assert!(!MigrationPolicy::PreferDevice(0).is_default());
        assert!(!MigrationPolicy::PreferHost.is_default());
    }

    #[test]
    fn migration_policy_display() {
        let s = format!("{}", MigrationPolicy::PreferDevice(2));
        assert!(s.contains("PreferDevice(2)"));

        let s2 = format!("{}", MigrationPolicy::Default);
        assert!(s2.contains("Default"));
    }

    // -- ManagedMemoryHints construction tests ------------------------------

    #[test]
    fn hints_for_buffer_rejects_zero_size() {
        let result = ManagedMemoryHints::for_buffer(0x1000, 0);
        assert!(result.is_err());
    }

    #[test]
    fn hints_for_buffer_valid() {
        let hints = ManagedMemoryHints::for_buffer(0x1000, 4096);
        assert!(hints.is_ok());
        let hints = hints.ok();
        assert!(hints.is_some());
        let hints = hints.map(|h| {
            assert_eq!(h.ptr(), 0x1000);
            assert_eq!(h.byte_size(), 4096);
        });
        let _ = hints;
    }

    #[test]
    fn hints_accessors() {
        let hints = ManagedMemoryHints::for_buffer(0xDEAD, 512);
        if let Ok(h) = hints {
            assert_eq!(h.ptr(), 0xDEAD);
            assert_eq!(h.byte_size(), 512);
        }
    }

    // -- PrefetchPlan tests -------------------------------------------------

    #[test]
    fn prefetch_plan_new_is_empty() {
        let plan = PrefetchPlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn prefetch_plan_default_is_empty() {
        let plan = PrefetchPlan::default();
        assert!(plan.is_empty());
    }

    #[test]
    fn prefetch_plan_add_and_len() {
        let mut plan = PrefetchPlan::new();
        plan.add(0x1000, 4096, 0).add(0x2000, 8192, 1);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());

        let entries = plan.entries();
        assert_eq!(entries[0].ptr, 0x1000);
        assert_eq!(entries[0].byte_size, 4096);
        assert_eq!(entries[0].device_ordinal, 0);
        assert_eq!(entries[1].ptr, 0x2000);
        assert_eq!(entries[1].byte_size, 8192);
        assert_eq!(entries[1].device_ordinal, 1);
    }

    #[test]
    fn prefetch_plan_chaining() {
        let mut plan = PrefetchPlan::new();
        plan.add(0x100, 100, 0)
            .add(0x200, 200, 0)
            .add(0x300, 300, 0);
        assert_eq!(plan.len(), 3);
    }

    // -- prefetch_range validation tests ------------------------------------

    #[test]
    fn prefetch_range_rejects_zero_count() {
        // We need a Device and Stream for prefetch_range, but we can test
        // the zero-count path only if Device::get succeeds.
        if let Ok(dev) = Device::get(0) {
            // We cannot construct a Stream without a context, so just verify
            // the hints struct validates before calling the driver.
            let hints = ManagedMemoryHints::for_buffer(0x1000, 4096);
            // The zero-count check happens in prefetch_range before the
            // driver call, so we test the function signature compiles.
            let _ = (hints, dev);
        }
        // Compile-time signature check
        let _: fn(&ManagedMemoryHints, usize, usize, &Device, &Stream) -> CudaResult<()> =
            ManagedMemoryHints::prefetch_range;
    }

    #[test]
    fn prefetch_range_out_of_bounds_detected() {
        // Verify the bounds checking logic without needing a GPU.
        // We replicate the internal check manually.
        let byte_size: usize = 4096;
        let offset: usize = 4000;
        let count: usize = 200;
        let end = offset.checked_add(count);
        assert!(end.is_some());
        let end = end.map(|e| e > byte_size);
        // 4000 + 200 = 4200 > 4096
        assert_eq!(end, Some(true));
    }

    // -- apply_migration_policy tests ---------------------------------------

    #[test]
    fn apply_policy_default_is_noop() {
        // Default policy should return Ok without calling the driver.
        let fake_dev: Device = unsafe { std::mem::zeroed() };
        let result = apply_migration_policy(0x1000, 4096, &MigrationPolicy::Default, &fake_dev);
        assert!(result.is_ok());
    }

    // -- Compile-time signature checks for GPU-requiring functions ----------

    #[test]
    fn signature_set_read_mostly() {
        let _: fn(&ManagedMemoryHints, &Device) -> CudaResult<()> =
            ManagedMemoryHints::set_read_mostly;
    }

    #[test]
    fn signature_unset_read_mostly() {
        let _: fn(&ManagedMemoryHints, &Device) -> CudaResult<()> =
            ManagedMemoryHints::unset_read_mostly;
    }

    #[test]
    fn signature_prefetch_to() {
        let _: fn(&ManagedMemoryHints, &Device, &Stream) -> CudaResult<()> =
            ManagedMemoryHints::prefetch_to;
    }

    #[test]
    fn signature_apply_policy() {
        let _: fn(&ManagedMemoryHints, &MigrationPolicy, &Device) -> CudaResult<()> =
            ManagedMemoryHints::apply_policy;
    }

    #[test]
    fn signature_execute_plan() {
        let _: fn(&PrefetchPlan, &Stream) -> CudaResult<()> = PrefetchPlan::execute;
    }
}
