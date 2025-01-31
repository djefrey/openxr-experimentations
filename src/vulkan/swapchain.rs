use std::{collections::BTreeMap, sync::Arc};

use glam::Mat4;
use openxr as xr;
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, descriptor_set::{layout::{DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType}, DescriptorSet, WriteDescriptorSet}, format::Format, image::{sys::RawImage, view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, pipeline::Pipeline, render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo}, shader::{DescriptorBindingRequirements, ShaderStages}, Handle, NonExhaustive};
use vulkano_macros::BufferContents;

use crate::openxr::XRState;

use ash::vk::Image as AshImage;

use super::VulkanState;

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct GlobalUniformData
{
    pub view: [Mat4; 2],
    pub proj: [Mat4; 2],
    pub proj_view: [Mat4; 2],
    pub inv_view: [Mat4; 2],
    pub inv_proj: [Mat4; 2],
}

struct SwapchainFrame
{
    framebuffer: Arc<Framebuffer>,
    color_frame: Arc<ImageView>,
    depth_frame: Arc<ImageView>
}

struct SwapchainGlobalUniformData
{
    data: Subbuffer<GlobalUniformData>,
    desc: Arc<DescriptorSet>
}

pub struct VulkanSwapchain
{
    pub handle: xr::Swapchain<xr::Vulkan>,
    frames: Vec<SwapchainFrame>,
    global_uniforms: Vec<SwapchainGlobalUniformData>,
    depth_desc_set: Vec<Arc<DescriptorSet>>,

    frame_idx: u32
}

impl VulkanSwapchain
{
    pub fn new(xr_state: &XRState, vk_state: &VulkanState) -> Self
    {
        let width  = xr_state.views[0].recommended_image_rect_width;
        let height = xr_state.views[0].recommended_image_rect_height;

        let swapchain_handle = xr_state.session.create_swapchain(&xr::SwapchainCreateInfo
        {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED,
            format: vulkano::format::Format::R8G8B8A8_SRGB as u32,
            sample_count: 1,
            width,
            height,
            face_count: 1,
            array_size: 2,
            mip_count: 1
        }).unwrap();

        let image_handles = swapchain_handle.enumerate_images().unwrap();
        let frames = image_handles.into_iter().map(|img_handle|
        {
            let img = unsafe
            {
                RawImage::from_handle(vk_state.device.clone(), AshImage::from_raw(img_handle.into()), ImageCreateInfo
                {
                    flags: ImageCreateFlags::empty(),
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    format: vulkano::format::Format::R8G8B8A8_SRGB,
                    samples: vulkano::image::SampleCount::Sample1,
                    extent: [width, height, 2],
                    mip_levels: 1,
                    array_layers: 2,
                    tiling: ImageTiling::Optimal,
                    image_type: ImageType::Dim2d,
                    ..Default::default()
                }).unwrap()
                .assume_bound() // session.create_swapchain already allocated the memory
            };

            let view = ImageView::new(Arc::new(img), ImageViewCreateInfo
            {
                format: vulkano::format::Format::R8G8B8A8_SRGB,
                view_type: vulkano::image::view::ImageViewType::Dim2dArray,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                subresource_range: ImageSubresourceRange
                {
                    aspects: ImageAspects::COLOR,
                    array_layers: 0..2,
                    mip_levels: 0..1,
                },
                ..Default::default()
            }).unwrap();

            let depth_img = Image::new(vk_state.allocators.std.clone(), ImageCreateInfo
            {
                format: Format::D32_SFLOAT,
                tiling: ImageTiling::Optimal,
                image_type: ImageType::Dim2d,
                extent: [width, height, 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                initial_layout: ImageLayout::Undefined,
                array_layers: 2,
                ..Default::default()
            }, AllocationCreateInfo
            {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            }).unwrap();

            let depth_view = ImageView::new(depth_img, ImageViewCreateInfo
            {
                format: Format::D32_SFLOAT,
                view_type: ImageViewType::Dim2dArray,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
                subresource_range: ImageSubresourceRange
                {
                    aspects: ImageAspects::DEPTH,
                    array_layers: 0..2,
                    mip_levels: 0..1,
                },
                ..Default::default()
            }).unwrap();

            let framebuffer = Framebuffer::new(vk_state.render_pass.clone(), FramebufferCreateInfo
            {
                flags: FramebufferCreateFlags::empty(),
                extent: [width, height],
                attachments: vec![view.clone(), depth_view.clone()],
                layers: 1,
                ..Default::default()
            }).unwrap();

            SwapchainFrame
            {
                framebuffer,
                color_frame: view,
                depth_frame: depth_view
            }
        }).collect::<Vec<_>>();

        let global_uniforms = frames.iter().map(|_|
        {
            let buffer = Buffer::new_sized::<GlobalUniformData>(vk_state.allocators.std.clone(),
                BufferCreateInfo
                {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo
                {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                }
            ).unwrap();

            let desc_layout = DescriptorSetLayout::new(vk_state.device.clone(), DescriptorSetLayoutCreateInfo
            {
                flags: DescriptorSetLayoutCreateFlags::empty(),
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::UniformBuffer,
                        )
                    },
                )].into_iter().collect(),
                ..Default::default()
            }).unwrap();

            let desc = DescriptorSet::new(
                vk_state.allocators.desc.clone(),
                desc_layout,
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                []
            ).unwrap();

            SwapchainGlobalUniformData
            {
                data: buffer,
                desc
            }
        }).collect::<Vec<_>>();

        let depth_desc_set = frames.iter().map(|frame|
        {
            DescriptorSet::new(
                vk_state.allocators.desc.clone(),
                vk_state.pipelines.cursor.layout().set_layouts()[1].clone(),
                [WriteDescriptorSet::image_view(0, frame.depth_frame.clone())],
                []
            ).unwrap()
        }).collect::<Vec<_>>();

        Self
        {
            handle: swapchain_handle,
            frames,
            global_uniforms,
            depth_desc_set,

            frame_idx: u32::MAX
        }
    }

    pub fn next_frame(&mut self)
    {
        self.frame_idx = self.handle.acquire_image().expect("Could not get next frame");
    }

    pub fn wait_frame(&mut self)
    {
        self.handle.wait_image(xr::Duration::INFINITE).expect("Error while waiting for frame");
    }

    pub fn release_frame(&mut self)
    {
        self.handle.release_image().expect("Error while releasing frame");
    }

    pub fn get_framebuffer(&self) -> &Arc<Framebuffer>
    {
        &self.frames[self.frame_idx as usize].framebuffer
    }

    pub fn get_color_frame(&self) -> &Arc<ImageView>
    {
        &self.frames[self.frame_idx as usize].color_frame
    }

    pub fn get_depth_frame(&self) -> &Arc<ImageView>
    {
        &self.frames[self.frame_idx as usize].depth_frame
    }

    pub fn update_global_unform(&self, data: &GlobalUniformData)
    {
        let buffer = &self.global_uniforms[self.frame_idx as usize].data;

        *buffer.write().unwrap() = *data;
    }

    pub fn get_global_uniform_descriptor(&self) -> &Arc<DescriptorSet>
    {
        &self.global_uniforms[self.frame_idx as usize].desc
    }

    pub fn get_depth_buffer_descriptor(&self) -> &Arc<DescriptorSet>
    {
        &self.depth_desc_set[self.frame_idx as usize]
    }
}
