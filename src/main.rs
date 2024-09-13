use core::f32;
use std::{f32::consts::PI, sync::Arc, time::Instant};

use glam::EulerRot;
use openxr::{CompositionLayerPassthroughFB, XRSetupState, XRState};
use vulkan::{BaseVertex, DepthUniformData, GlobalUniformData, LineVertex, ObjectData, VulkanState};

mod openxr;
mod vulkan;
mod obb;

use ::openxr::{self as xr, Duration, ViewConfigurationType};
use vulkano::{buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{allocator::{CommandBufferAllocator, StandardCommandBufferAllocator}, CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, CopyImageInfo, ImageCopy, RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo}, descriptor_set::{allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo}, DescriptorSet, WriteDescriptorSet}, format::{self, ClearValue}, image::{self, sampler::{BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode}, sys::RawImage, view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageMemory, ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage}, memory::{allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator}, DedicatedAllocation, ResourceMemory}, pipeline::{graphics::{depth_stencil::CompareOp, viewport::{Scissor, Viewport}}, Pipeline, PipelineBindPoint}, render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo}, sync::GpuFuture, Handle, NonExhaustive};
use obb::OBB;

struct MyFramebuffer
{
    handle: xr::Swapchain<xr::Vulkan>,
    frames: Vec<(Arc<Framebuffer>, Arc<ImageView>, Arc<ImageView>)>,
    global_uniforms: Vec<(Subbuffer<GlobalUniformData>, Arc<DescriptorSet>)>,
}

struct MyDepthSwapchain
{
    handle: xr::EnvironmentDepthSwapchain<xr::Vulkan>,
    data: Vec<(Subbuffer<DepthUniformData>, Arc<Sampler>, Arc<ImageView>, Arc<DescriptorSet>)>
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

#[derive(Clone, Copy)]
struct Transform
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

#[cfg(target_os = "android")]
fn request_spatial_permission()
{
    use jni::{objects::{JObject, JValue}, sys::jint};
    use ndk_glue::native_activity;

    let ctx = ndk_context::android_context();
    let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
    let activity = unsafe { JObject::from_raw(native_activity().activity()) };
    let mut env = vm.attach_current_thread().unwrap();

    let activity_class = env.find_class("android/app/Activity").unwrap();
    let permission = env.new_string("com.oculus.permission.USE_SCENE").unwrap();

    let permissions_array = env.new_object_array(1, "java/lang/String", JObject::null()).unwrap();
    env.set_object_array_element(&permissions_array, 0, permission).unwrap();

    let request_code : jint = 1;

    // let method_id = env.get_method_id(activity_class, , "([Ljava/lang/String;I)V").unwrap();
    env.call_method(activity, "requestPermissions", "([Ljava/lang/String;I)V", &[(&permissions_array).into(), request_code.into()]).unwrap();
}

#[cfg_attr(target_os = "android", ndk_glue::main)]
fn main()
{
    #[cfg(target_os = "android")]
    request_spatial_permission();

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
    let mut depth_swapchain : Option<MyDepthSwapchain> = None;

    let mut evt_storage = xr::EventDataBuffer::new();
    let mut running = false;
    let mut i = 0;

    let mut transform = Transform::new(glam::vec3(0.0, 1.5, 0.0), glam::Quat::IDENTITY, glam::vec3(0.2, 0.2, 0.2));
    let mut cube_obb = OBB::new(glam::vec3(0.0, 1.5, 0.0), glam::vec3(0.2, 0.2, 0.2), glam::Quat::IDENTITY);

    let mut hand : [Transform; 26] = [Transform::IDENTITY; 26];

    let mut wrist_last_frame : Option<Transform> = None;

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

        // let time_since_last_frame = Instant::now() - last_frame;
        // let rot_value = (PI / 2.0) * time_since_last_frame.as_secs_f32();

        // transform = transform.mul_mat4(&glam::Mat4::from_euler(EulerRot::XYX, rot_value, rot_value, 0.0));

        // xr_state.session.sync_actions(&[(&xr_state.action_set).into()]).unwrap();

        // let left_location = xr_state.left_space
        //     .locate(&xr_state.stage, frame_state.predicted_display_time)
        //     .unwrap();

        // let right_location = xr_state.right_space
        //     .locate(&xr_state.stage, frame_state.predicted_display_time)
        //     .unwrap();

        // println!("{:?} {:?}", left_location.pose.position, right_location.pose.position);

        #[cfg(target_os = "android")]
        {
        }

        // https://docs.unity3d.com/Packages/com.unity.xr.hands@1.4/manual/hand-data/xr-hand-data-model.html

        // if let Ok(hand_joint_maybeuninit) = xr_state.stage.locate_hand_joints(&xr_state.hand_tracker, frame_state.predicted_display_time)
        // {
        //     if let Some(hand_joint) = hand_joint_maybeuninit
        //     {
        //         let joint = hand_joint[xr::HandJoint::MIDDLE_PROXIMAL];
        //         let hand_pos = joint.pose.position;
        //         let hand_rot = joint.pose.orientation;

        //         transform = glam::Mat4::from_scale_rotation_translation(glam::vec3(0.1, 0.1, 0.1),
        //                                                                 glam::quat(hand_rot.x, hand_rot.y, hand_rot.z, hand_rot.w),
        //                                                                 glam::vec3(hand_pos.x, hand_pos.y, hand_pos.z));
        //     }
        // }

        if let Ok(hand_joint_maybeuninit) = xr_state.stage.locate_hand_joints(&xr_state.hand_tracker, frame_state.predicted_display_time)
        {
            if let Some(hand_joint) = hand_joint_maybeuninit
            {
                fn joint_to_transform(joint: xr::HandJointLocationEXT) -> Transform
                {
                    let pos = joint.pose.position;
                    let rot = joint.pose.orientation;

                    Transform
                    {
                        pos: glam::vec3(pos.x, pos.y, pos.z),
                        rot: glam::quat(rot.x, rot.y, rot.z, rot.w),
                        size: glam::vec3(0.03, 0.03, 0.03)
                    }
                }

                hand = hand_joint.map(joint_to_transform);

                {
                    let thumb_obb = OBB::from_transform(&hand[xr::HandJointEXT::THUMB_TIP]);
                    let mut cube_snapping = false;

                    if thumb_obb.does_collide_with(&cube_obb)
                    {
                        for tip in [xr::HandJointEXT::LITTLE_TIP, xr::HandJointEXT::RING_TIP, xr::HandJointEXT::MIDDLE_TIP, xr::HandJointEXT::INDEX_TIP]
                        {
                            let tip_obb = OBB::from_transform(&hand[tip]);

                            if tip_obb.does_collide_with(&cube_obb)
                            {
                                cube_snapping = true;
                                break;
                            }
                        }
                    }

                    if cube_snapping
                    {
                        let wrist = hand[xr::HandJointEXT::WRIST];

                        if let Some(last_wirst) = wrist_last_frame
                        {
                            let wrist_to_cube = transform.pos - wrist.pos;

                            let diff_pos = wrist.pos - last_wirst.pos;
                            let diff_rot = wrist.rot * last_wirst.rot.inverse();

                            transform.pos += diff_pos;
                            transform.rot = diff_rot.mul_quat(transform.rot);

                            transform.pos += diff_rot.mul_vec3(wrist_to_cube) - wrist_to_cube;

                            cube_obb.update_from_transform(&transform);
                        }

                        wrist_last_frame = Some(wrist);
                    }
                    else
                    {
                        wrist_last_frame = None;
                    }
                }
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

        let depth_swapchain = depth_swapchain.get_or_insert_with(||
        {
            let handle = xr_state.session
                .create_depth_environment_swapchain(&xr_state.depth_provider, xr::EnvironmentDepthSwapchainCreateFlagsMETA::EMPTY)
                .unwrap();

            let img_handles = handle.enumerate_images().unwrap();
            let state = handle.get_state().unwrap();

            let data = img_handles.into_iter().map(|img_handle|
            {
                let img = unsafe
                {
                    RawImage::from_handle(vk_state.device.clone(), ash::vk::Image::from_raw(img_handle), vulkano::image::ImageCreateInfo
                    {
                        flags: ImageCreateFlags::empty(),
                        usage: ImageUsage::SAMPLED,
                        image_type: ImageType::Dim2d,
                        format: vulkano::format::Format::D16_UNORM,
                        extent: [state.width, state.height, 1],
                        initial_layout: ImageLayout::ShaderReadOnlyOptimal,
                        mip_levels: 1,
                        array_layers: 2,
                        tiling: ImageTiling::Optimal,
                        ..Default::default()
                    }).unwrap().assume_bound()
                };

                let view = ImageView::new(Arc::new(img), ImageViewCreateInfo
                {
                    format: vulkano::format::Format::D16_UNORM,
                    view_type: vulkano::image::view::ImageViewType::Dim2dArray,
                    usage: ImageUsage::SAMPLED,
                    subresource_range: ImageSubresourceRange
                    {
                        aspects: ImageAspects::DEPTH,
                        array_layers: 0..2,
                        mip_levels: 0..1,
                    },
                    ..Default::default()
                }).unwrap();

                let buffer = Buffer::new_sized::<DepthUniformData>(allocator.clone(),
                    BufferCreateInfo
                    {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo
                    {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    }).unwrap();

                let sampler = Sampler::new(vk_state.device.clone(),
                    SamplerCreateInfo
                    {
                        address_mode: [SamplerAddressMode::ClampToBorder, SamplerAddressMode::ClampToBorder, SamplerAddressMode::ClampToBorder],
                        border_color: BorderColor::FloatOpaqueWhite,
                        min_filter: Filter::Linear,
                        mag_filter: Filter::Linear,
                        mipmap_mode: SamplerMipmapMode::Linear,
                        ..Default::default()
                    }).unwrap();

                let set = DescriptorSet::new(
                    desc_set_allocator.clone(),
                    vk_state.pipeline.layout().set_layouts()[1].clone(),
                    [
                        WriteDescriptorSet::buffer(0, buffer.clone()),
                        WriteDescriptorSet::image_view_sampler(1, view.clone(), sampler.clone()),
                    ],
                    [])
                .unwrap();

                return (buffer, sampler, view, set)
            }).collect::<Vec<_>>();

            MyDepthSwapchain { handle, data }
        });

        let view_to_matrix = |view: &xr::View, near: f32, far: f32| -> glam::Mat4
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

        let img_idx = swapchain.handle.acquire_image().unwrap() as usize;
        let depth_frame_data = xr_state.depth_provider.acquire_image(&xr_state.stage, frame_state.predicted_display_time).unwrap();

        *depth_swapchain.data[depth_frame_data.swapchain_index as usize].0.write().unwrap() = DepthUniformData
        {
            left: view_to_matrix(&depth_frame_data.views[0], depth_frame_data.near_z, depth_frame_data.far_z),
            right: view_to_matrix(&depth_frame_data.views[1], depth_frame_data.near_z, depth_frame_data.far_z),
        };

        // Writing Command Buffer

        i += 1;

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
            builder.push_constants(vk_state.pipeline.layout().clone(), 0, ObjectData
            {
                transform: transform.to_mat4(),
                tint: glam::vec4(1.0, 1.0, 1.0, 0.66),
            }).unwrap();

            builder.bind_vertex_buffers(0, [debug_cube_vertex_buffer.clone()]).unwrap()
                   .bind_index_buffer(indices_buffer.clone()).unwrap()
                   .bind_descriptor_sets(PipelineBindPoint::Graphics,
                        vk_state.pipeline.layout().clone(),
                        0,
                        vec![
                            swapchain.global_uniforms[img_idx].1.clone(),
                            depth_swapchain.data[depth_frame_data.swapchain_index as usize].3.clone()
                        ]).unwrap()
                   .draw_indexed(CUBE_INDICIES.len() as u32, 1, 0, 0, 0).unwrap();
        }

        unsafe
        {
            for tip in [xr::HandJointEXT::LITTLE_TIP, xr::HandJointEXT::RING_TIP, xr::HandJointEXT::MIDDLE_TIP, xr::HandJointEXT::INDEX_TIP, xr::HandJointEXT::THUMB_TIP]
            {
                let tip_obb = OBB::from_transform(&hand[tip]);
                let does_collide = tip_obb.does_collide_with(&cube_obb);

                builder.push_constants(vk_state.pipeline.layout().clone(), 0, ObjectData
                {
                    transform: hand[tip].to_mat4(),
                    tint: if does_collide { glam::vec4(0.0, 1.0, 0.0, 0.66) } else { glam::vec4(1.0, 0.0, 0.0, 0.66) },
                }).unwrap();

                builder.bind_vertex_buffers(0, [cube_vertex_buffer.clone()]).unwrap()
                       .bind_index_buffer(indices_buffer.clone()).unwrap()
                       .bind_descriptor_sets(PipelineBindPoint::Graphics,
                            vk_state.line_pipeline.layout().clone(),
                            0,
                            swapchain.global_uniforms[img_idx].1.clone()
                        ).unwrap()
                       .draw_indexed(CUBE_INDICIES.len() as u32, 1, 0, 0, 0).unwrap();
            }
        }

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

        builder.end_render_pass(Default::default()).unwrap();

        let cmd_buffer = builder.end().unwrap();

        // Post Command Buffer

        let (_, views) = xr_state.session.locate_views(ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &xr_state.stage).unwrap();

        // Global uniform buffer is updated at the last moment to use the most accurate view matrix possible

        let uniform_subbuffer = &swapchain.global_uniforms[img_idx].0;
        *uniform_subbuffer.write().unwrap() = GlobalUniformData
        {
            left: view_to_matrix(&views[0], 0.1, 100.0),
            right: view_to_matrix(&views[1], 0.1, 100.0)
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
