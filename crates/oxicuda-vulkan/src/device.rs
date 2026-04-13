//! Vulkan physical and logical device management.
//!
//! [`VulkanDevice`] owns the Vulkan `Entry`, `Instance`, and `Device` objects
//! and is responsible for their orderly destruction.  It selects the first
//! physical device that exposes a compute queue family.

use ash::{Device, Entry, Instance, vk};

use crate::error::{VulkanError, VulkanResult};

/// Owns the Vulkan instance and logical device together with the queue family
/// index and device memory properties needed for buffer allocation.
pub struct VulkanDevice {
    /// The ash entry point (holds the dynamically loaded Vulkan library).
    // NOTE: `entry` must be declared *before* `instance` so that the library
    // outlives the handles that depend on it.
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    compute_queue_family: u32,
    compute_queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
}

// SAFETY: Vulkan handles are valid to send across threads; the Vulkan spec
// guarantees that device and instance handles are externally synchronised by
// the caller (we guard mutable state with Mutex in higher-level structs).
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    /// Load the Vulkan library and create a device backed by the first
    /// physical device that has a compute queue family.
    ///
    /// Returns `Err(LibraryNotFound)` if no Vulkan driver is installed, and
    /// `Err(NoSuitableDevice)` if no compute-capable GPU is found.
    pub fn new() -> VulkanResult<Self> {
        // Runtime loading — no link-time dependency on libvulkan.
        let entry =
            unsafe { Entry::load() }.map_err(|e| VulkanError::LibraryNotFound(e.to_string()))?;

        let instance = Self::create_instance(&entry)?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| VulkanError::VkError(e.as_raw(), "enumerate_physical_devices".into()))?;

        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoSuitableDevice);
        }

        let (physical_device, compute_queue_family) =
            Self::select_device(&instance, &physical_devices)?;

        let device_name = Self::get_device_name(&instance, physical_device);

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let device = Self::create_logical_device(&instance, physical_device, compute_queue_family)
            .inspect_err(|_| {
                // On failure we must clean up the instance.
                unsafe { instance.destroy_instance(None) };
            })?;

        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        tracing::debug!(
            device = %device_name,
            compute_queue_family,
            "Vulkan device initialised"
        );

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            compute_queue_family,
            compute_queue,
            memory_properties,
            device_name,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn create_instance(entry: &Entry) -> VulkanResult<Instance> {
        let app_name = c"oxicuda-vulkan";
        let engine_name = c"oxicuda";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| VulkanError::VkError(e.as_raw(), format!("create_instance: {e}")))
    }

    fn select_device(
        instance: &Instance,
        devices: &[vk::PhysicalDevice],
    ) -> VulkanResult<(vk::PhysicalDevice, u32)> {
        // Prefer discrete GPU, then any device with a compute queue.
        let mut fallback: Option<(vk::PhysicalDevice, u32)> = None;

        for &dev in devices {
            let props = unsafe { instance.get_physical_device_properties(dev) };
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(dev) };

            for (i, qf) in queue_families.iter().enumerate() {
                if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    let qf_idx = i as u32;
                    if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        return Ok((dev, qf_idx));
                    }
                    if fallback.is_none() {
                        fallback = Some((dev, qf_idx));
                    }
                }
            }
        }

        fallback.ok_or(VulkanError::NoSuitableDevice)
    }

    fn get_device_name(instance: &Instance, physical_device: vk::PhysicalDevice) -> String {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        // device_name is [i8; 256] — convert to String safely.
        let name_bytes: Vec<u8> = props
            .device_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&name_bytes).into_owned()
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        compute_queue_family: u32,
    ) -> VulkanResult<Device> {
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(|e| VulkanError::VkError(e.as_raw(), format!("create_device: {e}")))
    }

    // ── Public accessors ─────────────────────────────────────────────────────

    /// Human-readable name of the selected physical device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Queue family index used for compute submissions.
    pub fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }

    /// The raw `vk::Queue` used for compute command submissions.
    pub fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    /// Reference to the logical device (used by memory manager, pipeline, etc.).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reference to the instance (used for extensions if needed).
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Reference to the physical device.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Find a memory type index that satisfies `type_filter` (bitmask from
    /// `vkGetBufferMemoryRequirements`) and has all `properties` flags set.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> VulkanResult<u32> {
        let mem_props = &self.memory_properties;
        for i in 0..mem_props.memory_type_count {
            let type_matches = (type_filter & (1 << i)) != 0;
            let prop_matches = mem_props.memory_types[i as usize]
                .property_flags
                .contains(properties);
            if type_matches && prop_matches {
                return Ok(i);
            }
        }
        Err(VulkanError::OutOfMemory)
    }

    /// Block until all submitted work on this device is complete.
    pub fn wait_idle(&self) -> VulkanResult<()> {
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| VulkanError::VkError(e.as_raw(), "device_wait_idle".into()))
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Destroy in reverse creation order: logical device first, then instance.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("device_name", &self.device_name)
            .field("compute_queue_family", &self.compute_queue_family)
            .finish()
    }
}
