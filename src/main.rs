use core::f32;
use std::{array, f32::consts::PI, sync::Arc, time::Instant};

use gestures::{Gesture, GestureKind, GesturePhase, GestureState, Hand, HandTip};
use glam::{vec3, EulerRot, Quat, Vec3, Vec4};
use object::{ObjectID, ObjectKind, ObjectList};
use openxr::{CompositionLayerPassthroughFB, XRSetupState, XRState};
use vulkan::{BaseVertex, GlobalUniformData, LineVertex, ObjectData, VulkanState};

mod openxr;
mod vulkan;
mod obb;
mod ray;
mod gestures;
mod object;

use ::openxr::{self as xr, Duration, ViewConfigurationType};
use vulkano::{buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{allocator::{CommandBufferAllocator, StandardCommandBufferAllocator}, CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo}, descriptor_set::{allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo}, DescriptorSet, WriteDescriptorSet}, format::{self, ClearValue}, image::{self, sys::RawImage, view::{ImageView, ImageViewCreateInfo, ImageViewType}, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageMemory, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage}, memory::{allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator}, DedicatedAllocation, ResourceMemory}, pipeline::{graphics::{depth_stencil::CompareOp, viewport::{Scissor, Viewport}}, Pipeline, PipelineBindPoint}, render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo}, sync::GpuFuture, Handle};
use obb::OBB;
use ray::Ray;

struct MyFramebuffer
{
    handle: xr::Swapchain<xr::Vulkan>,
    frames: Vec<(Arc<Framebuffer>, Arc<ImageView>, Arc<ImageView>)>,
    global_uniforms: Vec<(Subbuffer<GlobalUniformData>, Arc<DescriptorSet>)>,
}

const TRIANGLE_VERTICES : [BaseVertex; 3] = [
    BaseVertex { position: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0] },
    BaseVertex { position: [1.0, 0.0, 0.0], color: [0.0, 1.0, 0.0] },
    BaseVertex { position: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
];

const DEBUG_CUBE : [BaseVertex; 8] = [
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

const WHITE_CUBE : [BaseVertex; 8] = [
    // BOT
    BaseVertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0] },
    BaseVertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 1.0, 0.0] },
    BaseVertex { position: [-0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0] },
    BaseVertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0] },

    // TOP
    BaseVertex { position: [-0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0] },
    BaseVertex { position: [ 0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0] },
    BaseVertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0] },
    BaseVertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0] },
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


const HAND_LINES : [xr::HandJointEXT; 64] =
[
    xr::HandJointEXT::WRIST, xr::HandJointEXT::LITTLE_METACARPAL,
    xr::HandJointEXT::WRIST, xr::HandJointEXT::RING_METACARPAL,
    xr::HandJointEXT::WRIST, xr::HandJointEXT::MIDDLE_METACARPAL,
    xr::HandJointEXT::WRIST, xr::HandJointEXT::INDEX_METACARPAL,
    xr::HandJointEXT::WRIST, xr::HandJointEXT::THUMB_METACARPAL,

    xr::HandJointEXT::LITTLE_METACARPAL, xr::HandJointEXT::RING_METACARPAL,
    xr::HandJointEXT::RING_METACARPAL, xr::HandJointEXT::MIDDLE_METACARPAL,
    xr::HandJointEXT::MIDDLE_METACARPAL, xr::HandJointEXT::INDEX_METACARPAL,
    xr::HandJointEXT::INDEX_METACARPAL, xr::HandJointEXT::THUMB_METACARPAL,

    xr::HandJointEXT::LITTLE_METACARPAL, xr::HandJointEXT::LITTLE_PROXIMAL,
    xr::HandJointEXT::RING_METACARPAL, xr::HandJointEXT::RING_PROXIMAL,
    xr::HandJointEXT::MIDDLE_METACARPAL, xr::HandJointEXT::MIDDLE_PROXIMAL,
    xr::HandJointEXT::INDEX_METACARPAL, xr::HandJointEXT::INDEX_PROXIMAL,
    xr::HandJointEXT::THUMB_METACARPAL, xr::HandJointEXT::THUMB_PROXIMAL,

    xr::HandJointEXT::LITTLE_PROXIMAL, xr::HandJointEXT::RING_PROXIMAL,
    xr::HandJointEXT::RING_PROXIMAL, xr::HandJointEXT::MIDDLE_PROXIMAL,
    xr::HandJointEXT::MIDDLE_PROXIMAL, xr::HandJointEXT::INDEX_PROXIMAL,
    xr::HandJointEXT::INDEX_PROXIMAL, xr::HandJointEXT::THUMB_PROXIMAL,

    xr::HandJointEXT::LITTLE_PROXIMAL, xr::HandJointEXT::LITTLE_INTERMEDIATE,
    xr::HandJointEXT::LITTLE_INTERMEDIATE, xr::HandJointEXT::LITTLE_DISTAL,
    xr::HandJointEXT::LITTLE_DISTAL, xr::HandJointEXT::LITTLE_TIP,

    xr::HandJointEXT::RING_PROXIMAL, xr::HandJointEXT::RING_INTERMEDIATE,
    xr::HandJointEXT::RING_INTERMEDIATE, xr::HandJointEXT::RING_DISTAL,
    xr::HandJointEXT::RING_DISTAL, xr::HandJointEXT::RING_TIP,

    xr::HandJointEXT::MIDDLE_PROXIMAL, xr::HandJointEXT::MIDDLE_INTERMEDIATE,
    xr::HandJointEXT::MIDDLE_INTERMEDIATE, xr::HandJointEXT::MIDDLE_DISTAL,
    xr::HandJointEXT::MIDDLE_DISTAL, xr::HandJointEXT::MIDDLE_TIP,

    xr::HandJointEXT::INDEX_PROXIMAL, xr::HandJointEXT::INDEX_INTERMEDIATE,
    xr::HandJointEXT::INDEX_INTERMEDIATE, xr::HandJointEXT::INDEX_DISTAL,
    xr::HandJointEXT::INDEX_DISTAL, xr::HandJointEXT::INDEX_TIP,

    xr::HandJointEXT::THUMB_PROXIMAL, xr::HandJointEXT::THUMB_DISTAL,
    xr::HandJointEXT::THUMB_DISTAL, xr::HandJointEXT::THUMB_TIP,
];

#[derive(Debug, Clone, Copy)]
pub struct Transform
{
    pub pos: glam::Vec3,
    pub rot: glam::Quat,
    pub size: glam::Vec3
}

impl Transform
{
    pub const IDENTITY : Self = Self { pos: glam::Vec3::ZERO, rot: glam::Quat::IDENTITY, size: glam::Vec3::ONE };

    pub fn new(pos: glam::Vec3, rot: glam::Quat, size: glam::Vec3) -> Self
    {
        Self
        {
            pos,
            rot,
            size
        }
    }

    pub fn to_mat4(&self) -> glam::Mat4
    {
        glam::Mat4::from_scale_rotation_translation(self.size, self.rot, self.pos)
    }
}

#[cfg_attr(target_os = "android", ndk_glue::main)]
fn main()
{
    println!("MAAAAAAAIN");

    let setup_xr = XRSetupState::init().unwrap();
    let mut vk_state = VulkanState::from_xr(&setup_xr).unwrap();
    let mut xr_state = XRState::init(setup_xr, &vk_state).unwrap();

    let allocator = Arc::new(StandardMemoryAllocator::new_default(vk_state.device.clone()));
    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(vk_state.device.clone(), Default::default()));
    let desc_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(vk_state.device.clone(), StandardDescriptorSetAllocatorCreateInfo::default()));

    let debug_cube_vertex_buffer = Buffer::from_iter(allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        DEBUG_CUBE.into_iter())
    .unwrap();

    let cube_vertex_buffer = Buffer::from_iter(allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        WHITE_CUBE.into_iter())
    .unwrap();

    let hand_vertex_buffer = Buffer::from_iter(allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        HAND_LINES.map(|_| LineVertex { position: glam::Vec3::ZERO.to_array() }).into_iter())
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

    let raycast_buffer = Buffer::from_iter(allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![LineVertex { position: glam::Vec3::ZERO.to_array() }; 2].into_iter())
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

    let mut hand : Option<Hand> = None;

    let mut obj_list = ObjectList::new();
    let mut gestures = GestureState::new();

    let cube_ids : [ObjectID; 5] = array::from_fn(|i| obj_list.new_object(
        ObjectKind::DebugCube,
        Transform::new(vec3(0.0, 1.5, -1.0 - (i as f32) * 0.5), Quat::IDENTITY, vec3(0.3, 0.3, 0.3)),
        Some(OBB::CUBE_OBB)
    ));

    let tip_ids : [ObjectID; 5] = array::from_fn(|_|
    {
        obj_list.new_object(
            ObjectKind::TintedCube { tint: Vec4::ONE },
            Transform::new(Vec3::ZERO, Quat::IDENTITY, vec3(0.03, 0.03, 0.03)),
            None
        )
    });

    let show_debug = true;

    let mut last_frame : Instant = Instant::now();

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

        hand = Hand::from_xr_state(&xr_state, frame_state.predicted_display_time);

        let mut colliding_tips : [bool; 5] = [false; 5];

        for gesture in gestures.update(&hand, &obj_list)
        {
            println!("Gesture: {:?}", gesture);

            if gesture.phase != GesturePhase::Cancelled
            {
                match gesture.kind
                {
                    GestureKind::Grab { tips: _} | GestureKind::Ray =>
                    {
                        let obj = obj_list.get_mut_object(gesture.id).unwrap();

                        if let Some(diff) = gestures.transform_since_last_frame(&obj.transform.pos)
                        {
                            obj.transform.pos += diff.pos;
                            obj.transform.rot = diff.rot * obj.transform.rot;
                            // obj.transform.size *= diff.size;
                        }
                    },
                    _ => {}
                }
            }

            match gesture.kind
            {
                GestureKind::Tap { tips } | GestureKind::Grab { tips } =>
                {
                    for tip in tips
                    {
                        colliding_tips[tip.to_idx()] = true;
                    }
                },
                _ => {}
            }
        }

        // DEBUG
        if show_debug
        {
            for (i, &tip) in tip_ids.iter().enumerate()
            {
                let obj = obj_list.get_mut_object(tip).expect("Could not get tip");

                let tint = if colliding_tips[i] { Vec4::Y } else { Vec4::X };
                let new_transform = hand.as_ref().and_then(|hand| hand.get_tip(i)).unwrap_or(Transform::IDENTITY);

                obj.transform.pos = new_transform.pos;
                obj.transform.rot = new_transform.rot;
                obj.kind = ObjectKind::TintedCube { tint };
            }
        }

        // ---- Rendering -----

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
            let frames = image_handles.into_iter().map(|img_handle|
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
            }).collect::<Vec<_>>();

            let global_uniforms = frames.iter().map(|_|
            {
                let buffer = Buffer::new_sized::<GlobalUniformData>(allocator.clone(),
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

                let desc = DescriptorSet::new(
                    desc_set_allocator.clone(),
                    vk_state.pipeline.layout().set_layouts()[0].clone(),
                    [WriteDescriptorSet::buffer(0, buffer.clone())],
                    []
                ).unwrap();

                (buffer, desc)
            }).collect::<Vec<_>>();

            MyFramebuffer { handle: swapchain_handle, frames, global_uniforms }
        });

        let img_idx = swapchain.handle.acquire_image().unwrap() as usize;

        // Writing Command Buffer

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

        builder.bind_pipeline_graphics(vk_state.pipeline.clone()).unwrap();

        unsafe
        {
            for (id, obj) in obj_list.iter()
            {
                match obj.kind
                {
                    ObjectKind::DebugCube =>
                    {
                        builder.push_constants(vk_state.pipeline.layout().clone(), 0, ObjectData
                        {
                            transform: obj.transform.to_mat4(),
                            tint: glam::Vec4::ONE,
                        }).unwrap();

                        builder.bind_vertex_buffers(0, [debug_cube_vertex_buffer.clone()]).unwrap()
                               .bind_index_buffer(indices_buffer.clone()).unwrap()
                               .bind_descriptor_sets(PipelineBindPoint::Graphics, vk_state.pipeline.layout().clone(), 0, swapchain.global_uniforms[img_idx].1.clone()).unwrap()
                               .draw_indexed(CUBE_INDICIES.len() as u32, 1, 0, 0, 0).unwrap();
                    },

                    ObjectKind::TintedCube { tint } =>
                    {
                        builder.push_constants(vk_state.pipeline.layout().clone(), 0, ObjectData
                        {
                            transform: obj.transform.to_mat4(),
                            tint,
                        }).unwrap();

                        builder.bind_vertex_buffers(0, [cube_vertex_buffer.clone()]).unwrap()
                               .bind_index_buffer(indices_buffer.clone()).unwrap()
                               .bind_descriptor_sets(PipelineBindPoint::Graphics, vk_state.pipeline.layout().clone(), 0, swapchain.global_uniforms[img_idx].1.clone()).unwrap()
                               .draw_indexed(CUBE_INDICIES.len() as u32, 1, 0, 0, 0).unwrap();
                    },
                }
            }
        }

        // DEBUG

        if show_debug
        {
            if let Some(hand) = hand
            {
                builder.bind_pipeline_graphics(vk_state.line_pipeline.clone()).unwrap();

                unsafe
                {
                    (*hand_vertex_buffer.write().unwrap()).copy_from_slice(&HAND_LINES.map(|idx| LineVertex { position: hand[idx].pos.to_array() }));

                    builder.push_constants(vk_state.pipeline.layout().clone(), 0, ObjectData
                    {
                        transform: glam::Mat4::IDENTITY,
                        tint: glam::vec4(0.0, 0.0, 1.0, 0.0),
                    }).unwrap();

                    builder.bind_vertex_buffers(0, [hand_vertex_buffer.clone()]).unwrap()
                        .bind_descriptor_sets(PipelineBindPoint::Graphics, vk_state.pipeline.layout().clone(), 0, swapchain.global_uniforms[img_idx].1.clone()).unwrap()
                        .draw(HAND_LINES.len() as u32, 1, 0, 0).unwrap();
                }

                unsafe
                {
                    let ray = hand.compute_ray();

                    let does_intersects = 'block: {

                        for cube_id in cube_ids
                        {
                            let cube = obj_list.get_object(cube_id).expect("Could not get cube ID");
                            let cube_obb = OBB::CUBE_OBB.compute_obb(&cube.transform);

                            if cube_obb.does_intersects_ray(&ray).is_some()
                            {
                                break 'block true;
                            }
                        }

                        false
                    };

                    let color = if does_intersects { glam::vec4(0.0, 1.0, 0.0, 0.0) } else { glam::vec4(1.0, 0.0, 0.0, 0.0) };

                    (*raycast_buffer.write().unwrap()).copy_from_slice(&ray.to_points(5.0).map(|p| LineVertex { position: p.to_array() }));

                    builder.push_constants(vk_state.pipeline.layout().clone(), 0, ObjectData
                    {
                        transform: glam::Mat4::IDENTITY,
                        tint: color,
                    }).unwrap();

                    builder.bind_vertex_buffers(0, [raycast_buffer.clone()]).unwrap()
                        .bind_descriptor_sets(PipelineBindPoint::Graphics, vk_state.pipeline.layout().clone(), 0, swapchain.global_uniforms[img_idx].1.clone()).unwrap()
                        .draw(2, 1, 0, 0).unwrap();
                }
            }
        }

        builder.end_render_pass(Default::default()).unwrap();

        let cmd_buffer = builder.end().unwrap();

        // Post Command Buffer

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

        // Global uniform buffer is updated at the last moment to use the most accurate view matrix possible

        let uniform_subbuffer = &swapchain.global_uniforms[img_idx].0;
        *uniform_subbuffer.write().unwrap() = GlobalUniformData
        {
            left: view_to_matrix(&views[0]),
            right: view_to_matrix(&views[1])
        };

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

        last_frame = Instant::now();

        // println!("Frame displayed");
    }
}
