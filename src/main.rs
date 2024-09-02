use std::sync::Arc;

use openxr::{CompositionLayerPassthroughFB, XRSetupState, XRState};
use vulkan::{BaseVertex, VulkanState};

mod openxr;
mod vulkan;

use ::openxr::{self as xr, Duration, ViewConfigurationType};
use vulkano::{buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage}, command_buffer::{allocator::{CommandBufferAllocator, StandardCommandBufferAllocator}, CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo}, format::{self, ClearValue}, image::{self, sys::RawImage, view::{ImageView, ImageViewCreateInfo, ImageViewType}, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageMemory, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage}, memory::{allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator}, DedicatedAllocation, ResourceMemory}, pipeline::{graphics::{depth_stencil::CompareOp, viewport::{Scissor, Viewport}}, Pipeline}, render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo}, sync::GpuFuture, Handle};

struct MyFramebuffer
{
    handle: xr::Swapchain<xr::Vulkan>,
    frames: Vec<(Arc<Framebuffer>, Arc<ImageView>, Arc<ImageView>)>
}

const TRIANGLE_VERTICES : [BaseVertex; 3] = [
    BaseVertex { position: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0] },
    BaseVertex { position: [1.0, 0.0, 0.0], color: [0.0, 1.0, 0.0] },
    BaseVertex { position: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
];

const CUBE_VERTICES : [BaseVertex; 8] = [
    // BOT
    BaseVertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 0.0] },
    BaseVertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0] },
    BaseVertex { position: [-0.5, -0.5,  0.5], color: [0.0, 0.0, 1.0] },
    BaseVertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 0.0] },

    // TOP
    BaseVertex { position: [-0.5,  0.5, -0.5], color: [1.0, 0.0, 1.0] },
    BaseVertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 1.0] },
    BaseVertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0] },
    BaseVertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 0.0] },
];

const CUBE_INDICIES : [u16; 36] = [
    2, 6, 7,
    2, 7, 3,

    0, 5, 4,
    0, 1, 5,

    0, 6, 2,
    0, 4, 6,

    1, 3, 7,
    1, 7, 5,

    0, 2, 3,
    0, 3, 1,

    4, 7, 6,
    4, 5, 7,
];

#[repr(C)]
#[derive(BufferContents)]
struct ViewMatrices
{
    pub left : glam::Mat4,
    pub right : glam::Mat4
}

#[cfg_attr(target_os = "android", ndk_glue::main)]
fn main()
{
    let setup_xr = XRSetupState::init().unwrap();
    let mut vk_state = VulkanState::from_xr(&setup_xr).unwrap();
    let mut xr_state = XRState::init(setup_xr, &vk_state).unwrap();

    let allocator = Arc::new(StandardMemoryAllocator::new_default(vk_state.device.clone()));
    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(vk_state.device.clone(), Default::default()));

    let vertex_buffer = Buffer::from_iter(allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        CUBE_VERTICES.into_iter())
    .unwrap();

    let indices_buffer = Buffer::from_iter(allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        CUBE_INDICIES.into_iter())
    .unwrap();

    // let views_buffer = Buffer::new_sized::<ViewMatrices>(allocator.clone(),
    //     BufferCreateInfo {
    //         usage: BufferUsage::UNIFORM_BUFFER,
    //         ..Default::default()
    //     },
    //     AllocationCreateInfo {
    //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    //         ..Default::default()
    //     })
    // .unwrap();

    // views_buffer.write();

    let mut swapchain : Option<MyFramebuffer> = None;
    let mut evt_storage = xr::EventDataBuffer::new();
    let mut running = false;
    let mut i = 0;

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

        std::thread::sleep(std::time::Duration::from_millis(8));

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
            let frames : Vec<(Arc<Framebuffer>, Arc<ImageView>, Arc<ImageView>)> = image_handles.into_iter().map(|img_handle|
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

                let depth_img = image::Image::new(allocator.clone(), ImageCreateInfo
                {
                    format: format::Format::D32_SFLOAT,
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
                    format: format::Format::D32_SFLOAT,
                    view_type: ImageViewType::Dim2dArray,
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
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

                (framebuffer, view, depth_view)
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
                    clear_values: vec![Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])), Some(ClearValue::Depth(1.0))],
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

        swapchain.handle.wait_image(xr::Duration::INFINITE).unwrap();

        let (_, views) = xr_state.session.locate_views(ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &xr_state.stage).unwrap();

        let view_to_matrix = |view: &xr::View| -> glam::Mat4
        {
            let pos = view.pose.position;
            let rot = view.pose.orientation;
            let fov = view.fov;

            let view = glam::Mat4::from_rotation_translation(glam::quat(rot.x, rot.y, rot.z, rot.w), glam::vec3(pos.x, pos.y, pos.z)).inverse();            // angle_down is negative

            // https://github.com/KhronosGroup/OpenXR-SDK-Source/blob/4b9834dbf78f22f9a71500a13442c9bc2c7edb3c/src/common/xr_linear.h#L626

            let l = fov.angle_left.tan();
            let r = fov.angle_right.tan();
            let u = fov.angle_up.tan();
            let d = fov.angle_down.tan();

            let w = r - l;
            let h = d - u;

            let near_z = 0.1;
            let far_z = 100.0;

            let proj = glam::mat4(
                glam::vec4(2.0 / w,     0.0,         0.0,                                    0.0),
                glam::vec4(0.0,         2.0 / h,     0.0,                                    0.0),
                glam::vec4((r + l) / w, (u + d) / h, -far_z / (far_z - near_z),             -1.0),
                glam::vec4(0.0,         0.0,         -(far_z * near_z) / (far_z - near_z),   0.0)
            );

            return proj.mul_mat4(&view);
        };

        let view_matrices = ViewMatrices
        {
            left: view_to_matrix(&views[0]),
            right: view_to_matrix(&views[1])
        };

        i += 1;

        builder.push_constants(vk_state.pipeline.layout().clone(), 0, view_matrices).unwrap();


        // -------------
        // TO MOVE PRIOR WAITING IMG

        builder.bind_pipeline_graphics(vk_state.pipeline.clone()).unwrap();

        unsafe
        {
            builder.bind_vertex_buffers(0, [vertex_buffer.clone()]).unwrap()
                   .bind_index_buffer(indices_buffer.clone()).unwrap()
                   .draw_indexed(CUBE_INDICIES.len() as u32, 1, 0, 0, 0).unwrap();
        }

        builder.end_render_pass(Default::default()).unwrap();

        // -------------

        let cmd_buffer = builder.end().unwrap();

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
            #[cfg(target_os = "android")]
            &CompositionLayerPassthroughFB::new(&xr_state.passthrough.layer),
            &xr::CompositionLayerProjection::new()
                .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
                .space(&xr_state.stage)
                .views(&[
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
