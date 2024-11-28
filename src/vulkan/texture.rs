use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{BufferImageCopy, CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, CopyBufferToImageInfo, RecordingCommandBuffer}, descriptor_set::{DescriptorSet, WriteDescriptorSet}, format::Format, image::{sys::RawImage, view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageTiling, ImageType, ImageUsage}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, pipeline::Pipeline};

use super::VulkanState;

#[derive(Debug, Clone)]
pub struct VulkanTexture
{
    pub width: u32,
    pub height: u32,
    pub view: Arc<ImageView>,
    pub desc: Arc<DescriptorSet>
}

impl VulkanTexture
{
    // Probably suboptimal, buffer should be reused
    pub fn copy_data_to_texture(data: &[u8], img: &Arc<Image>, state: &VulkanState)
    {
        let buffer = Buffer::new_slice(state.allocators.std.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.len() as u64)
        .unwrap();

        (*buffer.write().unwrap()).copy_from_slice(data);

        let mut builder = RecordingCommandBuffer::new(state.allocators.cmd.clone(),
                                                      state.queue_family_index,
                                                      CommandBufferLevel::Primary,
                                                      CommandBufferBeginInfo{ usage: CommandBufferUsage::OneTimeSubmit, ..Default::default() }).unwrap();

        let cpy_info = CopyBufferToImageInfo::buffer_image(buffer, img.clone());

        println!("Buffer len: {}", data.len());
        println!("Cpy info: {:?}", cpy_info);

        builder.copy_buffer_to_image(cpy_info).unwrap();

        let cmd_buffer = builder.end().unwrap();

        cmd_buffer.execute(state.queue.clone()).unwrap();
    }

    pub fn new_rgb(width: u32, height: u32, data: &[u8], state: &VulkanState) -> Self
    {
        // println!("RGB FORMAT INFOS: {:?}", vulkano_pdevice.format_properties(Format::R8G8B8_UNORM));
        // println!("RGBA FORMAT INFOS: {:?}", vulkano_pdevice.format_properties(Format::R8G8B8A8_UNORM));

        // NOTE: R8G8B8_UNORM does not supports Optimal tiling (only R8G8B8A8_UNORM does)

        let img = Image::new(state.allocators.std.clone(), ImageCreateInfo
        {
            format: Format::R8G8B8_UNORM,
            tiling: ImageTiling::Linear,
            image_type: ImageType::Dim2d,
            extent: [width, height, 1],
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            initial_layout: ImageLayout::Undefined,
            array_layers: 1,
            ..Default::default()
        }, AllocationCreateInfo::default()).unwrap();

        VulkanTexture::copy_data_to_texture(data, &img, state);

        let view = ImageView::new_default(img).unwrap();

        let desc = DescriptorSet::new(state.allocators.desc.clone(),
                                      state.pipelines.textured.layout().set_layouts()[1].clone(),
                                      [WriteDescriptorSet::image_view_sampler(0, view.clone(), state.sampler.clone())],
                                      []).unwrap();

        Self
        {
            width,
            height,
            view,
            desc
        }
    }

    pub fn new_pixelated_rgb(width: u32, height: u32, data: &[u8], state: &VulkanState) -> Self
    {
        // println!("RGB FORMAT INFOS: {:?}", vulkano_pdevice.format_properties(Format::R8G8B8_UNORM));
        // println!("RGBA FORMAT INFOS: {:?}", vulkano_pdevice.format_properties(Format::R8G8B8A8_UNORM));

        // NOTE: R8G8B8_UNORM does not supports Optimal tiling (only R8G8B8A8_UNORM does)

        let img = Image::new(state.allocators.std.clone(), ImageCreateInfo
        {
            format: Format::R8G8B8_UNORM,
            tiling: ImageTiling::Linear,
            image_type: ImageType::Dim2d,
            extent: [width, height, 1],
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            initial_layout: ImageLayout::Undefined,
            array_layers: 1,
            ..Default::default()
        }, AllocationCreateInfo::default()).unwrap();

        VulkanTexture::copy_data_to_texture(data, &img, state);

        let view = ImageView::new_default(img).unwrap();

        let desc = DescriptorSet::new(state.allocators.desc.clone(),
                                      state.pipelines.textured.layout().set_layouts()[1].clone(),
                                      [WriteDescriptorSet::image_view_sampler(0, view.clone(), state.nearest_sampler.clone())],
                                      []).unwrap();

        Self
        {
            width,
            height,
            view,
            desc
        }
    }
}
