//! CUDA Driver API flag and constant definitions.
//!
//! Stream flags, event flags, memory pool attributes, memory attach flags,
//! host register flags, pointer attribute codes, memory type values, context
//! scheduling flags, function attribute constants, memory advise values,
//! limit constants, and occupancy flags.

// =========================================================================
// Stream creation flags
// =========================================================================

/// Default stream creation flag (implicit synchronisation with the NULL stream).
pub const CU_STREAM_DEFAULT: u32 = 0;

/// Stream does not synchronise with the NULL stream.
pub const CU_STREAM_NON_BLOCKING: u32 = 1;

// =========================================================================
// Stream-ordered memory pool attributes (CUDA 11.2+)
// =========================================================================

/// Pool reuse policy: follow event dependencies before reusing a freed block.
pub const CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: u32 = 1;

/// Pool reuse policy: allow opportunistic reuse without ordering guarantees.
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: u32 = 2;

/// Pool reuse policy: allow the driver to insert internal dependencies for reuse.
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: u32 = 3;

/// Release threshold (bytes): memory returned to OS when usage drops below this.
pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: u32 = 4;

/// Current reserved memory in bytes (read-only).
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: u32 = 5;

/// High-water mark of reserved memory in bytes (resettable).
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: u32 = 6;

/// Current used memory in bytes (read-only).
pub const CU_MEMPOOL_ATTR_USED_MEM_CURRENT: u32 = 7;

/// High-water mark of used memory in bytes (resettable).
pub const CU_MEMPOOL_ATTR_USED_MEM_HIGH: u32 = 8;

// =========================================================================
// Event creation flags
// =========================================================================

/// Default event creation flag.
pub const CU_EVENT_DEFAULT: u32 = 0;

/// Event uses blocking synchronisation.
pub const CU_EVENT_BLOCKING_SYNC: u32 = 1;

/// Event does not record timing data (faster).
pub const CU_EVENT_DISABLE_TIMING: u32 = 2;

/// Event may be used as an interprocess event.
pub const CU_EVENT_INTERPROCESS: u32 = 4;

// =========================================================================
// Memory-attach flags (for managed / mapped memory)
// =========================================================================

/// Memory is accessible from any stream on any device.
pub const CU_MEM_ATTACH_GLOBAL: u32 = 1;

/// Memory is initially accessible only from the allocating stream/host.
pub const CU_MEM_ATTACH_HOST: u32 = 2;

/// Memory is initially accessible only from a single stream.
pub const CU_MEM_ATTACH_SINGLE: u32 = 4;

// =========================================================================
// cuMemHostRegister flags
// =========================================================================

/// Registered memory is portable across CUDA contexts.
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;

/// Registered memory is mapped into the device address space.
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;

/// Pointer is to I/O memory (not system RAM).
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 0x04;

/// Registered memory will not be written by the GPU (read-only).
pub const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 0x08;

// =========================================================================
// cuPointerGetAttribute attribute codes
// =========================================================================

/// Query the CUDA context associated with a pointer.
pub const CU_POINTER_ATTRIBUTE_CONTEXT: u32 = 1;

/// Query the memory type (host / device / unified) of a pointer.
pub const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: u32 = 2;

/// Query the device pointer corresponding to a host pointer.
pub const CU_POINTER_ATTRIBUTE_DEVICE_POINTER: u32 = 3;

/// Query the host pointer corresponding to a device pointer.
pub const CU_POINTER_ATTRIBUTE_HOST_POINTER: u32 = 4;

/// Query whether the memory is managed (unified).
pub const CU_POINTER_ATTRIBUTE_IS_MANAGED: u32 = 7;

// =========================================================================
// CU_MEMORYTYPE values (returned by pointer attribute queries)
// =========================================================================

/// Host (system) memory.
pub const CU_MEMORYTYPE_HOST: u32 = 1;

/// Device (GPU) memory.
pub const CU_MEMORYTYPE_DEVICE: u32 = 2;

/// Array memory.
pub const CU_MEMORYTYPE_ARRAY: u32 = 3;

/// Unified (managed) memory.
pub const CU_MEMORYTYPE_UNIFIED: u32 = 4;

// =========================================================================
// Context scheduling flags
// =========================================================================

/// The driver picks the most appropriate scheduling mode.
pub const CU_CTX_SCHED_AUTO: u32 = 0;

/// Actively spin when waiting for results from the GPU.
pub const CU_CTX_SCHED_SPIN: u32 = 1;

/// Yield the CPU when waiting for results from the GPU.
pub const CU_CTX_SCHED_YIELD: u32 = 2;

/// Block the calling thread when waiting for results.
pub const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 4;

/// Mask for the scheduling flags.
pub const CU_CTX_SCHED_MASK: u32 = 0x07;

/// Support mapped pinned allocations.
pub const CU_CTX_MAP_HOST: u32 = 0x08;

/// Keep local memory allocation after launch.
pub const CU_CTX_LMEM_RESIZE_TO_MAX: u32 = 0x10;

/// Coredump enable.
pub const CU_CTX_COREDUMP_ENABLE: u32 = 0x20;

/// User coredump enable.
pub const CU_CTX_USER_COREDUMP_ENABLE: u32 = 0x40;

/// Sync-memops flag.
pub const CU_CTX_SYNC_MEMOPS: u32 = 0x80;

/// Mask for all context flags.
pub const CU_CTX_FLAGS_MASK: u32 = 0xFF;

// =========================================================================
// Function attribute values (used with cuFuncGetAttribute)
// =========================================================================

/// Maximum threads per block for this function.
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 0;

/// Shared memory used by this function (bytes).
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: i32 = 1;

/// Size of user-allocated constant memory (bytes).
pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: i32 = 2;

/// Size of local memory used by each thread (bytes).
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: i32 = 3;

/// Number of registers used by each thread.
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: i32 = 4;

/// PTX virtual architecture version (e.g. 70 for sm_70).
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION: i32 = 5;

/// Binary architecture version (e.g. 70 for sm_70).
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION: i32 = 6;

/// Whether this function has been cached.
pub const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: i32 = 7;

/// Maximum dynamic shared memory size (bytes).
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

/// Preferred shared memory carve-out.
pub const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: i32 = 9;

/// Cluster size setting.
pub const CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: i32 = 10;

/// Required cluster width.
pub const CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: i32 = 11;

/// Required cluster height.
pub const CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: i32 = 12;

/// Required cluster depth.
pub const CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: i32 = 13;

/// Non-portable cluster size allowed.
pub const CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: i32 = 14;

/// Required cluster scheduling policy preference.
pub const CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: i32 = 15;

// =========================================================================
// Memory advise values
// =========================================================================

/// Hint that the data will be read mostly.
pub const CU_MEM_ADVISE_SET_READ_MOSTLY: u32 = 1;

/// Unset read-mostly hint.
pub const CU_MEM_ADVISE_UNSET_READ_MOSTLY: u32 = 2;

/// Set the preferred location to the specified device.
pub const CU_MEM_ADVISE_SET_PREFERRED_LOCATION: u32 = 3;

/// Unset the preferred location.
pub const CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: u32 = 4;

/// Set access from the specified device.
pub const CU_MEM_ADVISE_SET_ACCESSED_BY: u32 = 5;

/// Unset access from the specified device.
pub const CU_MEM_ADVISE_UNSET_ACCESSED_BY: u32 = 6;

// =========================================================================
// Limit values (cuCtxSetLimit / cuCtxGetLimit)
// =========================================================================

/// Stack size for each GPU thread.
pub const CU_LIMIT_STACK_SIZE: u32 = 0;

/// Size of the printf FIFO.
pub const CU_LIMIT_PRINTF_FIFO_SIZE: u32 = 1;

/// Size of the heap used by `malloc()` on the device.
pub const CU_LIMIT_MALLOC_HEAP_SIZE: u32 = 2;

/// Maximum nesting depth of a device runtime launch.
pub const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: u32 = 3;

/// Maximum number of outstanding device runtime launches.
pub const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: u32 = 4;

/// L2 cache fetch granularity.
pub const CU_LIMIT_MAX_L2_FETCH_GRANULARITY: u32 = 5;

/// Maximum persisting L2 cache size.
pub const CU_LIMIT_PERSISTING_L2_CACHE_SIZE: u32 = 6;

// =========================================================================
// Occupancy flags
// =========================================================================

/// Default occupancy calculation.
pub const CU_OCCUPANCY_DEFAULT: u32 = 0;

/// Disable caching override.
pub const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE: u32 = 1;

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_and_event_flags() {
        assert_eq!(CU_STREAM_DEFAULT, 0);
        assert_eq!(CU_STREAM_NON_BLOCKING, 1);
        assert_eq!(CU_EVENT_DEFAULT, 0);
        assert_eq!(CU_EVENT_BLOCKING_SYNC, 1);
        assert_eq!(CU_EVENT_DISABLE_TIMING, 2);
        assert_eq!(CU_EVENT_INTERPROCESS, 4);
    }

    #[test]
    fn test_context_scheduling_flags() {
        assert_eq!(CU_CTX_SCHED_AUTO, 0);
        assert_eq!(CU_CTX_SCHED_SPIN, 1);
        assert_eq!(CU_CTX_SCHED_YIELD, 2);
        assert_eq!(CU_CTX_SCHED_BLOCKING_SYNC, 4);
    }

    #[test]
    fn test_mem_attach_flags() {
        assert_eq!(CU_MEM_ATTACH_GLOBAL, 1);
        assert_eq!(CU_MEM_ATTACH_HOST, 2);
        assert_eq!(CU_MEM_ATTACH_SINGLE, 4);
    }

    #[test]
    fn test_func_attribute_constants() {
        assert_eq!(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0);
        assert_eq!(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 1);
        assert_eq!(CU_FUNC_ATTRIBUTE_NUM_REGS, 4);
    }

    #[test]
    fn test_limit_constants() {
        assert_eq!(CU_LIMIT_STACK_SIZE, 0);
        assert_eq!(CU_LIMIT_PRINTF_FIFO_SIZE, 1);
        assert_eq!(CU_LIMIT_MALLOC_HEAP_SIZE, 2);
    }

    #[test]
    fn test_memory_type_constants() {
        assert_eq!(CU_MEMORYTYPE_HOST, 1);
        assert_eq!(CU_MEMORYTYPE_DEVICE, 2);
        assert_eq!(CU_MEMORYTYPE_ARRAY, 3);
        assert_eq!(CU_MEMORYTYPE_UNIFIED, 4);
    }
}
