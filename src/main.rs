use std::sync::Arc;

use openxr::{XRSetupState, XRState};
use vulkan::VulkanState;

mod openxr;
mod vulkan;

use ::openxr::{self as xr, Duration, ViewConfigurationType};
use vulkano::{command_buffer::{allocator::{CommandBufferAllocator, StandardCommandBufferAllocator}, CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo}, image::{self, sys::RawImage, view::{ImageView, ImageViewCreateInfo}, ImageAspects, ImageCreateFlags, ImageMemory, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage}, memory::{allocator::{AllocationCreateInfo, MemoryAllocator, StandardMemoryAllocator}, DedicatedAllocation, ResourceMemory}, pipeline::graphics::viewport::{Scissor, Viewport}, render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo}, sync::GpuFuture, Handle};

struct MyFramebuffer
{
    handle: xr::Swapchain<xr::Vulkan>,
    frames: Vec<(Arc<Framebuffer>, Arc<ImageView>)>
}

fn main()
{
    let setup_xr = XRSetupState::init().unwrap();
    let mut vk_state = VulkanState::from_xr(&setup_xr).unwrap();
    let mut xr_state = XRState::init(setup_xr, &vk_state).unwrap();

    let allocator = Arc::new(StandardMemoryAllocator::new_default(vk_state.device.clone()));
    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(vk_state.device.clone(), Default::default()));

    let mut swapchain : Option<MyFramebuffer> = None;
    let mut evt_storage = xr::EventDataBuffer::new();
    let mut running = false;

'main_loop: loop
    {
        while let Some(evt) = xr_state.instance.poll_event(&mut evt_storage).unwrap()
        {
            use xr::Event::*;

            match evt
            {
                SessionStateChanged(e) =>
                {
                    println!("Entering state: {:?}", e.state());

                    match e.state()
                    {
                        xr::SessionState::READY =>
                        {
                            xr_state.session.begin(ViewConfigurationType::PRIMARY_STEREO).unwrap();
                            running = true;
                        },
                        xr::SessionState::STOPPING =>
                        {
                            xr_state.session.end().unwrap();
                            running = false;
                        },
                        xr::SessionState::EXITING =>
                        {
                            break 'main_loop;
                        },
                        _ => {}
                    }
                },

                InstanceLossPending(_) =>
                {
                    break 'main_loop;
                },

                _ => {},
            }
        }

        if !running
        {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let frame_state = xr_state.frame_waiter.wait().unwrap();

        xr_state.frame_stream.begin().unwrap();

        if !frame_state.should_render
        {
            xr_state.frame_stream.end(frame_state.predicted_display_time, xr_state.environment_blend_mode, &[]).unwrap();
            continue;
        }

        let width  = xr_state.views[0].recommended_image_rect_width;
        let height = xr_state.views[0].recommended_image_rect_height;

        let swapchain = swapchain.get_or_insert_with(||
        {
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
            let frames : Vec<(Arc<Framebuffer>, Arc<ImageView>)> = image_handles.into_iter().map(|img_handle|
            {
                let img = unsafe
                {
                    RawImage::from_handle(vk_state.device.clone(), ash::vk::Image::from_raw(img_handle), vulkano::image::ImageCreateInfo
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

                let framebuffer = Framebuffer::new(vk_state.render_pass.clone(), FramebufferCreateInfo
                {
                    flags: FramebufferCreateFlags::empty(),
                    extent: [width, height],
                    attachments: vec![view.clone()],
                    layers: 1,
                    ..Default::default()
                }).unwrap();

                (framebuffer, view)
            }).collect();

            MyFramebuffer { handle: swapchain_handle, frames: frames }
        });

        let img_idx = swapchain.handle.acquire_image().unwrap() as usize;

        // println!("Render in framebuffer #{}: {} {}", img_idx, width, height);

        let mut builder = RecordingCommandBuffer::new(cmd_allocator.clone(),
                                                      vk_state.queue_family_index,
                                                      CommandBufferLevel::Primary,
                                                      CommandBufferBeginInfo{ usage: CommandBufferUsage::OneTimeSubmit, ..Default::default() }).unwrap();

        builder.begin_render_pass(
                RenderPassBeginInfo
                {
                    clear_values: vec![Some([0.0, 1.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(swapchain.frames[img_idx].0.clone())
                },
                SubpassBeginInfo
                {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                }
            ).unwrap()
            .set_viewport(0, [Viewport { offset: [0.0, 0.0], extent: [width as f32, height as f32], depth_range: 0.0..=1.0 }].into_iter().collect()).unwrap()
            .set_scissor(0, [Scissor { offset: [0, 0], extent: [width, height] }].into_iter().collect()).unwrap();

        unsafe
        {
            builder.bind_pipeline_graphics(vk_state.pipeline.clone()).unwrap()
                .draw(3, 1, 0, 0).unwrap();
        }

        builder.end_render_pass(Default::default()).unwrap();

        let cmd_buffer = builder.end().unwrap();

        let (_, views) = xr_state.session.locate_views(ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &xr_state.stage).unwrap();

        swapchain.handle.wait_image(xr::Duration::INFINITE).unwrap();

        let _ = cmd_buffer.execute(vk_state.queue.clone()).unwrap()
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

        swapchain.handle.release_image().unwrap();

        let rect = xr::Rect2Di
        {
            offset: xr::Offset2Di { x: 0, y: 0},
            extent: xr::Extent2Di { width: width as i32, height: height as i32 }
        };

        xr_state.frame_stream.end(frame_state.predicted_display_time, xr_state.environment_blend_mode, &[
            &xr::CompositionLayerProjection::new().space(&xr_state.stage).views(&[
                xr::CompositionLayerProjectionView::new()
                    .pose(views[0].pose)
                    .fov(views[0].fov)
                    .sub_image(xr::SwapchainSubImage::new()
                        .swapchain(&swapchain.handle)
                        .image_array_index(0)
                        .image_rect(rect)
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(views[1].pose)
                    .fov(views[1].fov)
                    .sub_image(xr::SwapchainSubImage::new()
                        .swapchain(&swapchain.handle)
                        .image_array_index(1)
                        .image_rect(rect)
                    )
            ]),
        ]).unwrap();

        // println!("Frame displayed");
    }
}
