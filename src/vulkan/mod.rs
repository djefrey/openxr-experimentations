use std::{ffi::CStr, mem::transmute, sync::Arc};

use buffer::{BufferContents, Subbuffer};
use buffers::{LineVertex, TintedObjectData, VulkanBuffers};
use command_buffer::{allocator::StandardCommandBufferAllocator, RecordingCommandBuffer};
use descriptor_set::{allocator::StandardDescriptorSetAllocator, DescriptorSet};
use glam::vec4;
use image::{sampler::{self, Sampler, SamplerAddressMode, SamplerCreateInfo}, sys::RawImage, view::{ImageView, ImageViewType}, Image, ImageAspects, ImageCreateInfo, ImageFormatInfo, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage};
use memory::allocator::StandardMemoryAllocator;
use pipeline::{graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{DepthState, DepthStencilState}, input_assembly::{InputAssemblyState, PrimitiveTopology}, multisample::MultisampleState, rasterization::{CullMode, FrontFace, RasterizationState}, vertex_input::{Vertex, VertexDefinition}, viewport::ViewportState, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use pipelines::VulkanPipelines;
use swapchain::VulkanSwapchain;
use texture::VulkanTexture;
use vulkano::{*, format::Format, instance::*, device::*, device::physical::*, render_pass::*};
use ash::vk::{self, Handle};

use crate::{object::{Object, ObjectKind}, openxr::{XRSetupState, XRState}, Transform};

pub mod buffers;
pub mod pipelines;
pub mod swapchain;
pub mod texture;

pub struct VulkanAllocators
{
    pub std: Arc<StandardMemoryAllocator>,
    pub cmd: Arc<StandardCommandBufferAllocator>,
    pub desc: Arc<StandardDescriptorSetAllocator>,
}

pub struct VulkanState
{
    pub instance: Arc<Instance>,
    pub pdevice: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub queue_family_index: u32,
    pub render_pass: Arc<RenderPass>,
    pub sampler: Arc<Sampler>,
    pub nearest_sampler: Arc<Sampler>,
    pub allocators: VulkanAllocators,
    pub buffers: VulkanBuffers,
    pub pipelines: VulkanPipelines,
}

impl VulkanState
{
    pub fn from_xr(xr: &XRSetupState) -> Option<Self>
    {
        unsafe
        {
            let entry = ash::Entry::load().ok()?;
            let instance_proc_addr : openxr::sys::platform::VkGetInstanceProcAddr = transmute(entry.static_fn().get_instance_proc_addr);

            let app_info = vk::ApplicationInfo::default()
                .application_name(&CStr::from_bytes_with_nul_unchecked(b"OpenXR Tests\0"))
                .api_version(vk::API_VERSION_1_3);

            // Instance

            let instance_create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info);

            let instance_create_info_ptr : *const vk::InstanceCreateInfo = &instance_create_info;
            let instance_ptr = xr.instance.create_vulkan_instance(xr.system_id, instance_proc_addr, instance_create_info_ptr as *const _)
                .ok()?.ok()? as openxr::sys::platform::VkInstance;
            let instance = ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(instance_ptr as _));

            // Physcial Device

            let pdevice_ptr = xr.instance.vulkan_graphics_device(xr.system_id, instance_ptr as _)
                .ok()? as openxr::sys::platform::VkPhysicalDevice;
            let pdevice = vk::PhysicalDevice::from_raw(pdevice_ptr as _);

            // Queue Family Index

            let queue_family_index = instance.get_physical_device_queue_family_properties(pdevice)
                .into_iter()
                .enumerate()
                .position(|(_, info)|
                {
                    info.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
                })
                .map(|i| i as u32)?;

            let mut multview_feature = vk::PhysicalDeviceMultiviewFeatures
            {
                multiview: vk::TRUE,
                ..Default::default()
            };

            let queue_create_infos = [
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&[1.0]),
            ];

            let device_create_info = vk::DeviceCreateInfo::default()
                .push_next(&mut multview_feature)
                .queue_create_infos(&queue_create_infos);

            let device_create_info_ptr : *const vk::DeviceCreateInfo = &device_create_info;
            let device_ptr = xr.instance.create_vulkan_device(xr.system_id, instance_proc_addr, pdevice_ptr, device_create_info_ptr as *const _)
                .ok()?.ok()? as openxr::sys::platform::VkDevice;
            let device = ash::Device::load(instance.fp_v1_0(), vk::Device::from_raw(device_ptr as _));

            let vulkano_library = VulkanLibrary::with_loader(VulkanEntryWrapper(entry)).ok()?;
            let vulkano_instance = Instance::from_handle(vulkano_library, instance.handle(), InstanceCreateInfo
            {
                application_name: Some("openxr_tests".to_string()),
                ..Default::default()
            });
            let vulkano_pdevice = PhysicalDevice::from_handle(vulkano_instance.clone(), pdevice).ok()?;

            let mut features = DeviceFeatures::default();
            features.multiview = true;

            let (vulkano_device, mut queues) = Device::from_handle(vulkano_pdevice.clone(), device.handle(), DeviceCreateInfo
            {
                queue_create_infos: vec![
                    QueueCreateInfo
                    {
                        queue_family_index,
                        queues: vec![1.0],
                        ..Default::default()
                    },
                ],
                enabled_features: features,
                ..Default::default()
            });

            let queue = queues.next()?;

            let render_pass = RenderPass::new(vulkano_device.clone(), RenderPassCreateInfo
            {
                attachments: vec![AttachmentDescription
                {
                    format: format::Format::R8G8B8A8_SRGB,
                    samples: image::SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    initial_layout: image::ImageLayout::Undefined,
                    final_layout: image::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
                AttachmentDescription
                {
                    format: format::Format::D32_SFLOAT,
                    samples: image::SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    stencil_load_op: Some(AttachmentLoadOp::DontCare),
                    stencil_store_op: Some(AttachmentStoreOp::DontCare),
                    initial_layout: image::ImageLayout::Undefined,
                    final_layout: image::ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
                    ..Default::default()
                }],
                subpasses: vec![
                    SubpassDescription
                    {
                        color_attachments: vec![Some(AttachmentReference
                        {
                            attachment: 0,
                            layout: image::ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        })],
                        depth_stencil_attachment: Some(AttachmentReference
                        {
                            attachment: 1,
                            layout: image::ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
                            ..Default::default()
                        }),
                        view_mask: 0b11,
                        ..Default::default()
                    },
                ],
                dependencies: vec![SubpassDependency
                {
                    dst_subpass: Some(0),
                    src_stages: sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT | sync::PipelineStages::EARLY_FRAGMENT_TESTS,
                    dst_stages: sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT | sync::PipelineStages::EARLY_FRAGMENT_TESTS,
                    dst_access: sync::AccessFlags::COLOR_ATTACHMENT_WRITE | sync::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ..Default::default()
                }],
                correlated_view_masks: vec![0b11],
                ..Default::default()
            }).unwrap();

            let sampler = Sampler::new( vulkano_device.clone(), SamplerCreateInfo
                {
                    mag_filter: sampler::Filter::Linear,
                    min_filter: sampler::Filter::Linear,
                    mipmap_mode: sampler::SamplerMipmapMode::Nearest,
                    address_mode: [SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge],
                    mip_lod_bias: 0.0,
                    ..Default::default()
                }).unwrap();

            let nearest_sampler = Sampler::new( vulkano_device.clone(), SamplerCreateInfo
            {
                mag_filter: sampler::Filter::Nearest,
                min_filter: sampler::Filter::Nearest,
                mipmap_mode: sampler::SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge],
                mip_lod_bias: 0.0,
                ..Default::default()
            }).unwrap();

            let allocators = VulkanAllocators::new(&vulkano_device);
            let buffers = VulkanBuffers::new(&allocators);
            let pipelines = VulkanPipelines::new(&vulkano_device, &render_pass);

            Some(VulkanState
            {
                instance: vulkano_instance,
                pdevice: vulkano_pdevice,
                device: vulkano_device,
                queue,
                queue_family_index,
                render_pass,
                sampler,
                nearest_sampler,
                allocators,
                buffers,
                pipelines,
            })
        }
    }
}

impl VulkanAllocators
{
    pub fn new(device: &Arc<Device>) -> Self
    {
        let std = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cmd = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));
        let desc = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));

        Self
        {
            std,
            cmd,
            desc,
        }
    }
}

impl VulkanState
{
    pub unsafe fn render_debug_cube(&self, obj: &Object, swapchain: &VulkanSwapchain, builder: &mut RecordingCommandBuffer)
    {
        let pipeline = &self.pipelines.tinted;
        let layout = pipeline.layout();
        let vertex = &self.buffers.debug_cube_vertex;
        let index = &self.buffers.cube_index;

        builder.bind_pipeline_graphics(pipeline.clone()).unwrap();

        builder.push_constants(layout.clone(), 0, TintedObjectData
        {
            transform: obj.transform.to_mat4(),
            tint: glam::Vec4::ONE,
        }).unwrap();

        builder.bind_vertex_buffers(0, [vertex.clone()]).unwrap()
               .bind_index_buffer(index.clone()).unwrap()
               .bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0, swapchain.get_global_uniform_descriptor().clone()).unwrap()
               .draw_indexed(index.len() as u32, 1, 0, 0, 0).unwrap();
    }

    pub unsafe fn render_tinted_cube(&self, obj: &Object, tint: &glam::Vec4, swapchain: &VulkanSwapchain, builder: &mut RecordingCommandBuffer)
    {
        let pipeline = &self.pipelines.tinted;
        let layout = pipeline.layout();
        let vertex = &self.buffers.cube_vertex;
        let index = &self.buffers.cube_index;

        builder.bind_pipeline_graphics(pipeline.clone()).unwrap();

        builder.push_constants(layout.clone(), 0, TintedObjectData
        {
            transform: obj.transform.to_mat4(),
            tint: *tint,
        }).unwrap();

        builder.bind_vertex_buffers(0, [vertex.clone()]).unwrap()
               .bind_index_buffer(index.clone()).unwrap()
               .bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0, swapchain.get_global_uniform_descriptor().clone()).unwrap()
               .draw_indexed(index.len() as u32, 1, 0, 0, 0).unwrap();
    }

    pub unsafe fn render_textured_quad(&self, obj: &Object, texture: &VulkanTexture, swapchain: &VulkanSwapchain, builder: &mut RecordingCommandBuffer)
    {
        let pipeline = &self.pipelines.textured;
        let layout = pipeline.layout();
        let vertex = &self.buffers.quad_vertex;
        let index = &self.buffers.quad_index;

        builder.bind_pipeline_graphics(pipeline.clone()).unwrap();

        builder.push_constants(layout.clone(), 0, TintedObjectData
        {
            transform: obj.transform.to_mat4(),
            tint: vec4(1.0, 1.0, 1.0, 1.0),
        }).unwrap();

        builder.bind_vertex_buffers(0, [vertex.clone()]).unwrap()
               .bind_index_buffer(index.clone()).unwrap()
               .bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0,
                vec![
                    swapchain.get_global_uniform_descriptor().clone(),
                    texture.desc.clone(),
                ]).unwrap()
               .draw_indexed(index.len() as u32, 1, 0, 0, 0).unwrap();
    }

    pub unsafe fn render_wireframe(&self, buffer: &Subbuffer<[LineVertex]>, tint: &glam::Vec4, swapchain: &VulkanSwapchain, builder: &mut RecordingCommandBuffer)
    {
        let pipeline = &self.pipelines.line;
        let layout = pipeline.layout();

        builder.bind_pipeline_graphics(pipeline.clone()).unwrap();

        builder.push_constants(layout.clone(), 0, TintedObjectData
        {
            transform: glam::Mat4::IDENTITY,
            tint: *tint,
        }).unwrap();

        builder.bind_vertex_buffers(0, [buffer.clone()]).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0, swapchain.get_global_uniform_descriptor().clone()).unwrap()
            .draw(buffer.len() as u32, 1, 0, 0).unwrap();
    }
}

struct VulkanEntryWrapper(ash::Entry);

unsafe impl vulkano::library::Loader for VulkanEntryWrapper
{
    unsafe fn get_instance_proc_addr(&self, instance: ash::vk::Instance, name: *const std::os::raw::c_char) -> ash::vk::PFN_vkVoidFunction
    {
        return (self.0.static_fn().get_instance_proc_addr)(instance, name as *const _);
    }
}
