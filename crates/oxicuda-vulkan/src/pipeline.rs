//! Vulkan compute pipeline setup.
//!
//! [`VulkanComputePipeline`] creates and owns the Vulkan objects needed to
//! dispatch a compute shader:
//!
//! * `vk::ShaderModule` — compiled SPIR-V binary.
//! * `vk::DescriptorSetLayout` — layout describing the storage-buffer bindings.
//! * `vk::PipelineLayout` — wraps the descriptor set layout.
//! * `vk::Pipeline` — the compiled compute pipeline.
//! * `vk::DescriptorPool` — pool from which descriptor sets are allocated.
//!
//! Objects are destroyed in reverse creation order in `Drop`.

use ash::vk;
use std::sync::Arc;

use crate::device::VulkanDevice;
use crate::error::{VulkanError, VulkanResult};

/// A compiled Vulkan compute pipeline together with its descriptor
/// infrastructure.
pub struct VulkanComputePipeline {
    device: Arc<VulkanDevice>,
    shader_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
}

// SAFETY: VulkanDevice is Send+Sync and all Vulkan handles are opaque integers
// that are safe to send across thread boundaries when properly synchronised.
unsafe impl Send for VulkanComputePipeline {}
unsafe impl Sync for VulkanComputePipeline {}

impl VulkanComputePipeline {
    /// Create a compute pipeline from the given SPIR-V `spv_words` slice.
    ///
    /// * `spv_words` — SPIR-V binary as a slice of `u32` words.
    /// * `bindings` — number of storage-buffer descriptor bindings (binding
    ///   indices 0..bindings-1).
    /// * `max_sets` — number of descriptor sets to pre-allocate in the pool.
    pub fn new(
        device: Arc<VulkanDevice>,
        spv_words: &[u32],
        bindings: u32,
        max_sets: u32,
    ) -> VulkanResult<Self> {
        let vk_dev = device.device();

        // 1. Shader module
        let shader_module = {
            // SAFETY: `spv_words` is a valid SPIR-V binary (caller's responsibility).
            let code = unsafe { std::slice::from_raw_parts(spv_words.as_ptr(), spv_words.len()) };
            let create_info = vk::ShaderModuleCreateInfo::default().code(code);
            unsafe { vk_dev.create_shader_module(&create_info, None) }
                .map_err(|e| VulkanError::ShaderError(format!("create_shader_module: {e}")))?
        };

        // 2. Descriptor set layout — N storage buffers at consecutive binding points.
        let descriptor_set_layout = {
            let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);

            unsafe { vk_dev.create_descriptor_set_layout(&layout_info, None) }.map_err(|e| {
                unsafe { vk_dev.destroy_shader_module(shader_module, None) };
                VulkanError::PipelineError(format!("descriptor_set_layout: {e}"))
            })?
        };

        // 3. Pipeline layout
        let pipeline_layout = {
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            unsafe { vk_dev.create_pipeline_layout(&layout_info, None) }.map_err(|e| {
                unsafe {
                    vk_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    vk_dev.destroy_shader_module(shader_module, None);
                };
                VulkanError::PipelineError(format!("pipeline_layout: {e}"))
            })?
        };

        // 4. Compute pipeline
        let pipeline = {
            let entry_name = c"main";
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_name);

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let result = unsafe {
                vk_dev.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
            };

            match result {
                Ok(pipelines) => pipelines.into_iter().next().ok_or_else(|| {
                    unsafe {
                        vk_dev.destroy_pipeline_layout(pipeline_layout, None);
                        vk_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                        vk_dev.destroy_shader_module(shader_module, None);
                    };
                    VulkanError::PipelineError("no pipeline returned".into())
                })?,
                Err((_, e)) => {
                    unsafe {
                        vk_dev.destroy_pipeline_layout(pipeline_layout, None);
                        vk_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                        vk_dev.destroy_shader_module(shader_module, None);
                    };
                    return Err(VulkanError::PipelineError(format!(
                        "create_compute_pipelines: {e}"
                    )));
                }
            }
        };

        // 5. Descriptor pool
        let descriptor_pool = {
            let pool_sizes = if bindings > 0 {
                vec![vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: bindings * max_sets,
                }]
            } else {
                vec![]
            };

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(max_sets.max(1))
                .pool_sizes(&pool_sizes);

            unsafe { vk_dev.create_descriptor_pool(&pool_info, None) }.map_err(|e| {
                unsafe {
                    vk_dev.destroy_pipeline(pipeline, None);
                    vk_dev.destroy_pipeline_layout(pipeline_layout, None);
                    vk_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    vk_dev.destroy_shader_module(shader_module, None);
                };
                VulkanError::PipelineError(format!("descriptor_pool: {e}"))
            })?
        };

        Ok(Self {
            device,
            shader_module,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
        })
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// The compute pipeline handle.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// The pipeline layout handle.
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// The descriptor set layout handle.
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    /// The descriptor pool handle.
    pub fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }
}

impl Drop for VulkanComputePipeline {
    fn drop(&mut self) {
        let vk_dev = self.device.device();
        // Destroy in reverse creation order.
        unsafe {
            vk_dev.destroy_descriptor_pool(self.descriptor_pool, None);
            vk_dev.destroy_pipeline(self.pipeline, None);
            vk_dev.destroy_pipeline_layout(self.pipeline_layout, None);
            vk_dev.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vk_dev.destroy_shader_module(self.shader_module, None);
        }
    }
}

impl std::fmt::Debug for VulkanComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanComputePipeline")
            .field("pipeline", &self.pipeline)
            .finish()
    }
}
