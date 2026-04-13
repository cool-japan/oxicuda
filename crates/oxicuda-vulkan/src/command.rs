//! Vulkan command pool and command buffer submission helpers.
//!
//! [`VulkanCommandPool`] owns a `vk::CommandPool` allocated for the compute
//! queue family and provides `record_and_submit` for single-shot command
//! execution followed by a host-side wait (suitable for synchronous compute
//! dispatch in this backend).

use ash::vk;
use std::sync::Arc;

use crate::device::VulkanDevice;
use crate::error::{VulkanError, VulkanResult};

/// Owns a `vk::CommandPool` and provides convenience methods for recording
/// and submitting compute commands.
pub struct VulkanCommandPool {
    device: Arc<VulkanDevice>,
    command_pool: vk::CommandPool,
}

// SAFETY: VulkanDevice is Send+Sync and command pool handles are safe to send
// across threads when properly synchronised.
unsafe impl Send for VulkanCommandPool {}
unsafe impl Sync for VulkanCommandPool {}

impl VulkanCommandPool {
    /// Create a command pool for the device's compute queue family.
    pub fn new(device: Arc<VulkanDevice>) -> VulkanResult<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.compute_queue_family())
            // Allow individual command buffers to be reset.
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.device().create_command_pool(&pool_info, None) }
            .map_err(|e| VulkanError::CommandBufferError(format!("create_command_pool: {e}")))?;

        Ok(Self {
            device,
            command_pool,
        })
    }

    /// Allocate a single primary command buffer, call `record_fn` to fill it,
    /// submit it to the compute queue, and block until it completes.
    ///
    /// A `vk::Fence` is used to wait for completion so that the host is not
    /// spin-waiting.
    pub fn record_and_submit<F>(&self, record_fn: F) -> VulkanResult<()>
    where
        F: FnOnce(vk::CommandBuffer) -> VulkanResult<()>,
    {
        let vk_dev = self.device.device();

        // 1. Allocate one command buffer.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_bufs = unsafe { vk_dev.allocate_command_buffers(&alloc_info) }.map_err(|e| {
            VulkanError::CommandBufferError(format!("allocate_command_buffers: {e}"))
        })?;
        let cmd_buf = cmd_bufs[0];

        // 2. Begin recording.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { vk_dev.begin_command_buffer(cmd_buf, &begin_info) }
            .map_err(|e| VulkanError::CommandBufferError(format!("begin_command_buffer: {e}")))?;

        // 3. Record user commands.
        let record_result = record_fn(cmd_buf);

        // End recording regardless of whether the record closure succeeded.
        let end_result = unsafe { vk_dev.end_command_buffer(cmd_buf) }
            .map_err(|e| VulkanError::CommandBufferError(format!("end_command_buffer: {e}")));

        // Report recording errors before submission errors.
        record_result?;
        end_result?;

        // 4. Create a fence for synchronisation.
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { vk_dev.create_fence(&fence_info, None) }
            .map_err(|e| VulkanError::CommandBufferError(format!("create_fence: {e}")))?;

        // 5. Submit.
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));

        let submit_result = unsafe {
            vk_dev.queue_submit(
                self.device.compute_queue(),
                std::slice::from_ref(&submit_info),
                fence,
            )
        }
        .map_err(|e| VulkanError::CommandBufferError(format!("queue_submit: {e}")));

        // 6. Wait (5-second timeout).
        let wait_result = if submit_result.is_ok() {
            unsafe { vk_dev.wait_for_fences(std::slice::from_ref(&fence), true, 5_000_000_000) }
                .map_err(|e| VulkanError::CommandBufferError(format!("wait_for_fences: {e}")))
        } else {
            Ok(())
        };

        // 7. Clean up fence and command buffer regardless of errors.
        unsafe {
            vk_dev.destroy_fence(fence, None);
            vk_dev.free_command_buffers(self.command_pool, std::slice::from_ref(&cmd_buf));
        }

        submit_result?;
        wait_result?;

        Ok(())
    }

    /// Return the underlying `vk::CommandPool` handle.
    pub fn raw(&self) -> vk::CommandPool {
        self.command_pool
    }
}

impl Drop for VulkanCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

impl std::fmt::Debug for VulkanCommandPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanCommandPool")
            .field("command_pool", &self.command_pool)
            .finish()
    }
}
