//! Multi-queue async compute manager for overlapping compute and transfer.
//!
//! [`AsyncComputeManager`] manages multiple compute queues discovered during
//! device initialisation.  It provides round-robin queue selection and
//! per-queue fence-based synchronisation so that independent dispatches can
//! overlap on hardware that exposes more than one compute queue.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ash::vk;

use crate::command::VulkanCommandPool;
use crate::device::VulkanDevice;
use crate::error::{VulkanError, VulkanResult};

// ── Synchronisation primitives ──────────────────────────────────────────────

/// A Vulkan fence wrapper for host-side synchronisation.
pub struct VulkanFence {
    device: Arc<VulkanDevice>,
    fence: vk::Fence,
}

// SAFETY: Vulkan fence handles are safe to send across threads when externally
// synchronised (our API ensures single-ownership semantics).
unsafe impl Send for VulkanFence {}
unsafe impl Sync for VulkanFence {}

impl VulkanFence {
    /// Create a new unsignalled fence.
    pub fn new(device: Arc<VulkanDevice>) -> VulkanResult<Self> {
        let info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.device().create_fence(&info, None) }
            .map_err(|e| VulkanError::CommandBufferError(format!("create_fence: {e}")))?;
        Ok(Self { device, fence })
    }

    /// Create a fence that starts in the signalled state.
    pub fn new_signalled(device: Arc<VulkanDevice>) -> VulkanResult<Self> {
        let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.device().create_fence(&info, None) }
            .map_err(|e| VulkanError::CommandBufferError(format!("create_fence: {e}")))?;
        Ok(Self { device, fence })
    }

    /// Block until this fence is signalled (5-second timeout).
    pub fn wait(&self) -> VulkanResult<()> {
        unsafe {
            self.device.device().wait_for_fences(
                std::slice::from_ref(&self.fence),
                true,
                5_000_000_000,
            )
        }
        .map_err(|e| VulkanError::CommandBufferError(format!("wait_for_fences: {e}")))
    }

    /// Reset this fence to the unsignalled state.
    pub fn reset(&self) -> VulkanResult<()> {
        unsafe {
            self.device
                .device()
                .reset_fences(std::slice::from_ref(&self.fence))
        }
        .map_err(|e| VulkanError::CommandBufferError(format!("reset_fences: {e}")))
    }

    /// Raw Vulkan fence handle.
    pub fn raw(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for VulkanFence {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_fence(self.fence, None);
        }
    }
}

impl std::fmt::Debug for VulkanFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanFence")
            .field("fence", &self.fence)
            .finish()
    }
}

/// A Vulkan semaphore wrapper for inter-queue synchronisation.
pub struct VulkanSemaphore {
    device: Arc<VulkanDevice>,
    semaphore: vk::Semaphore,
}

// SAFETY: Vulkan semaphore handles are safe to send across threads when
// externally synchronised.
unsafe impl Send for VulkanSemaphore {}
unsafe impl Sync for VulkanSemaphore {}

impl VulkanSemaphore {
    /// Create a new binary semaphore.
    pub fn new(device: Arc<VulkanDevice>) -> VulkanResult<Self> {
        let info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { device.device().create_semaphore(&info, None) }
            .map_err(|e| VulkanError::CommandBufferError(format!("create_semaphore: {e}")))?;
        Ok(Self { device, semaphore })
    }

    /// Create a timeline semaphore (Vulkan 1.2+) with the given initial value.
    pub fn new_timeline(device: Arc<VulkanDevice>, initial_value: u64) -> VulkanResult<Self> {
        let mut timeline_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        timeline_info.initial_value = initial_value;

        let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);

        let semaphore = unsafe { device.device().create_semaphore(&info, None) }.map_err(|e| {
            VulkanError::CommandBufferError(format!("create_timeline_semaphore: {e}"))
        })?;
        Ok(Self { device, semaphore })
    }

    /// Signal a timeline semaphore to the given value (Vulkan 1.2+).
    pub fn signal(&self, value: u64) -> VulkanResult<()> {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);
        unsafe { self.device.device().signal_semaphore(&signal_info) }
            .map_err(|e| VulkanError::CommandBufferError(format!("signal_semaphore: {e}")))
    }

    /// Wait for a timeline semaphore to reach the given value (Vulkan 1.2+).
    ///
    /// Blocks for up to 5 seconds.
    pub fn wait(&self, value: u64) -> VulkanResult<()> {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.semaphore))
            .values(std::slice::from_ref(&value));
        unsafe {
            self.device
                .device()
                .wait_semaphores(&wait_info, 5_000_000_000)
        }
        .map_err(|e| VulkanError::CommandBufferError(format!("wait_semaphores: {e}")))
    }

    /// Raw Vulkan semaphore handle.
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for VulkanSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_semaphore(self.semaphore, None);
        }
    }
}

impl std::fmt::Debug for VulkanSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanSemaphore")
            .field("semaphore", &self.semaphore)
            .finish()
    }
}

// ── AsyncComputeManager ─────────────────────────────────────────────────────

/// Per-queue state: a command pool and a fence for tracking in-flight work.
struct QueueSlot {
    command_pool: VulkanCommandPool,
    fence: VulkanFence,
    queue: vk::Queue,
}

/// Manages multiple compute queues for overlapping async dispatch.
///
/// Each queue gets its own command pool and fence.  `next_queue()` cycles
/// through queues in round-robin order.
pub struct AsyncComputeManager {
    device: Arc<VulkanDevice>,
    slots: Vec<QueueSlot>,
    next: AtomicUsize,
}

// SAFETY: All interior fields are Send+Sync; AtomicUsize provides lock-free
// synchronisation for the round-robin counter.
unsafe impl Send for AsyncComputeManager {}
unsafe impl Sync for AsyncComputeManager {}

impl AsyncComputeManager {
    /// Create the manager, allocating a command pool and fence per queue.
    pub fn new(device: Arc<VulkanDevice>) -> VulkanResult<Self> {
        let queues = device.compute_queues().to_vec();
        let mut slots = Vec::with_capacity(queues.len());

        for &queue in &queues {
            let command_pool = VulkanCommandPool::new(Arc::clone(&device))?;
            let fence = VulkanFence::new_signalled(Arc::clone(&device))?;
            slots.push(QueueSlot {
                command_pool,
                fence,
                queue,
            });
        }

        Ok(Self {
            device,
            slots,
            next: AtomicUsize::new(0),
        })
    }

    /// Number of available compute queues.
    pub fn queue_count(&self) -> usize {
        self.slots.len()
    }

    /// Return the next queue index using round-robin scheduling.
    pub fn next_queue(&self) -> usize {
        if self.slots.is_empty() {
            return 0;
        }
        self.next.fetch_add(1, Ordering::Relaxed) % self.slots.len()
    }

    /// Submit a recorded command buffer to a specific queue.
    ///
    /// The `record_fn` closure receives a `vk::CommandBuffer` to record into.
    /// Submission is asynchronous: the function returns as soon as the GPU
    /// begins processing.  Call [`wait_all`](Self::wait_all) to drain.
    pub fn submit_async<F>(&self, queue_index: usize, record_fn: F) -> VulkanResult<()>
    where
        F: FnOnce(vk::CommandBuffer) -> VulkanResult<()>,
    {
        let slot = self
            .slots
            .get(queue_index)
            .ok_or(VulkanError::QueueIndexOutOfBounds(
                queue_index,
                self.slots.len(),
            ))?;

        // Wait for previous work on this queue to complete before reuse.
        slot.fence.wait()?;
        slot.fence.reset()?;

        let vk_dev = self.device.device();

        // Allocate a one-shot command buffer.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(slot.command_pool.raw())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_bufs = unsafe { vk_dev.allocate_command_buffers(&alloc_info) }.map_err(|e| {
            VulkanError::CommandBufferError(format!("allocate_command_buffers: {e}"))
        })?;
        let cmd_buf = cmd_bufs[0];

        // Begin recording.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { vk_dev.begin_command_buffer(cmd_buf, &begin_info) }
            .map_err(|e| VulkanError::CommandBufferError(format!("begin_command_buffer: {e}")))?;

        // Record user commands.
        let record_result = record_fn(cmd_buf);

        let end_result = unsafe { vk_dev.end_command_buffer(cmd_buf) }
            .map_err(|e| VulkanError::CommandBufferError(format!("end_command_buffer: {e}")));

        record_result?;
        end_result?;

        // Submit with fence for later synchronisation.
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));

        unsafe {
            vk_dev.queue_submit(
                slot.queue,
                std::slice::from_ref(&submit_info),
                slot.fence.raw(),
            )
        }
        .map_err(|e| VulkanError::CommandBufferError(format!("queue_submit: {e}")))?;

        Ok(())
    }

    /// Wait for all queues to drain their in-flight work.
    pub fn wait_all(&self) -> VulkanResult<()> {
        for slot in &self.slots {
            slot.fence.wait()?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for AsyncComputeManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncComputeManager")
            .field("queue_count", &self.slots.len())
            .field("next", &self.next.load(Ordering::Relaxed))
            .finish()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn async_manager_queue_count() {
        // If Vulkan is not available, skip.
        let device = match VulkanDevice::new() {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };
        let manager = AsyncComputeManager::new(Arc::clone(&device)).expect("create async manager");
        // Must have at least 1 queue (the primary compute queue).
        assert!(manager.queue_count() >= 1);
        assert_eq!(manager.queue_count(), device.compute_queue_count());
    }

    #[test]
    fn async_not_initialized() {
        // Without a Vulkan device we cannot create an AsyncComputeManager.
        // If Vulkan IS available, we verify that queue_index bounds are checked.
        let device = match VulkanDevice::new() {
            Ok(d) => Arc::new(d),
            Err(_) => {
                // No Vulkan -- the error itself proves the "not initialized" path.
                return;
            }
        };
        let manager = AsyncComputeManager::new(Arc::clone(&device)).expect("create async manager");
        // Submitting to an out-of-bounds queue must fail.
        let result = manager.submit_async(usize::MAX, |_cb| Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn round_robin_cycles() {
        let device = match VulkanDevice::new() {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };
        let manager = AsyncComputeManager::new(Arc::clone(&device)).expect("create async manager");
        let count = manager.queue_count();
        // Cycle through 3 full rounds and verify wrap-around.
        for round in 0..3 {
            for expected in 0..count {
                let got = manager.next_queue();
                assert_eq!(
                    got, expected,
                    "round {round}, expected queue {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn wait_all_without_queues_is_noop() {
        // Construct a manager with a real device; wait_all on freshly-created
        // (signalled) fences should return immediately.
        let device = match VulkanDevice::new() {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };
        let manager = AsyncComputeManager::new(Arc::clone(&device)).expect("create async manager");
        manager.wait_all().expect("wait_all on idle manager");
    }
}
