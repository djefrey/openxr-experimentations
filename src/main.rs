use core::f32;
use std::{array, sync::{Arc, Mutex}, thread, time::Instant};

use gestures::{GestureKind, GesturePhase, GestureState, Hand};
use glam::{vec3, vec4, Quat, Vec3, Vec4};
use object::{ObjectID, ObjectKind, ObjectList};
use openxr::{CompositionLayerPassthroughFB, XRSetupState, XRState};
use ray::Ray;
use vulkan::{swapchain::{self, GlobalUniformData, VulkanSwapchain}, texture::VulkanTexture, RenderState, VulkanState};

mod openxr;
mod vulkan;
mod obb;
mod ray;
mod gestures;
mod object;
mod window;

use ::openxr::{self as xr, ViewConfigurationType};
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{allocator::StandardCommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, CopyBufferToImageInfo, RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents}, descriptor_set::{allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo}, DescriptorSet, WriteDescriptorSet}, format::{self, ClearValue}, image::{self, sys::RawImage, view::{ImageView, ImageViewCreateInfo, ImageViewType}, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage}, memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{graphics::viewport::{Scissor, Viewport}, Pipeline, PipelineBindPoint}, render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo}, sync::GpuFuture, Handle};
use obb::OBB;
use window::{Window, WindowEvent};
use communication_lib::{tcp_lib::TcpConnection, udp_lib::UdpCommunication};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct WindowSize
{
    pub width: u32,
    pub height: u32,
}

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
    let setup_xr = XRSetupState::init().unwrap();
    let mut vk_state = VulkanState::from_xr(&setup_xr).unwrap();
    let mut xr_state = XRState::init(setup_xr, &vk_state).unwrap();

    let mut evt_storage = xr::EventDataBuffer::new();
    let mut running = false;

    let mut hand : Option<Hand> = None;

    let mut swapchain: Option<VulkanSwapchain> = None;
    let mut obj_list = ObjectList::new();
    let mut gestures = GestureState::new();

    let cube_ids : [ObjectID; 5] = array::from_fn(|i| obj_list.new_object(
        ObjectKind::DebugCube,
        Transform::new(vec3(0.0, 1.5, -1.0 - (i as f32) * 0.5), Quat::IDENTITY, vec3(0.3, 0.3, 0.3)),
        Some(OBB::CUBE_OBB)
    ));

    let bmp_data : [u8; 27] = [
        // TOP
        255,   0,   0,
          0, 255,   0,
          0,   0, 255,

        // MIDDLE
        255, 255,  0,
        255,   0, 255,
        0,   255, 255,

        // BOTTOM
          0,   0,  0,
        255, 255, 255,
        127, 127, 127

    ];

    // let quad_id = obj_list.new_object(ObjectKind::TexturedQuad
    //     {
    //         texture: VulkanTexture::new_pixelated_rgb(3, 3, &bmp_data, &vk_state)
    //     },
    //     Transform::new(vec3(2.0, 1.5, -1.0), Quat::IDENTITY, vec3(1.0, 1.0, 1.0)),
    //     Some(OBB::new(Vec3::new(1.0, 1.0, 0.001)))
    // );

    let window_id = Window::new(
        "Test Window".to_string(),
        VulkanTexture::new_pixelated_rgb(3, 3, &bmp_data, &vk_state),
        vec3(0.0, 0.5, 0.0), Quat::IDENTITY,
        &mut obj_list
    );

    let tip_ids : [ObjectID; 5] = array::from_fn(|_|
    {
        obj_list.new_object(
            ObjectKind::TintedCube { tint: Vec4::ONE },
            Transform::new(Vec3::ZERO, Quat::IDENTITY, vec3(0.03, 0.03, 0.03)),
            None
        )
    });

    let hand_id = obj_list.new_object(ObjectKind::Hand
        {
            hand: Hand::IDENTITY,
            buffer: vk_state.buffers.new_hand_wireframe_buffer()
        },
        Transform::IDENTITY,
        None
    );

    let raycast_id = obj_list.new_object(ObjectKind::Raycast
        {
            ray: Ray::X,
            buffer: vk_state.buffers.new_raycast_buffer()
        },
        Transform::IDENTITY,
        None
    );

    let show_debug = true;

    let mut last_frame : Instant = Instant::now();

    // NOTE: ALWAYS LOCK IN THIS ORDER
    let vk_state_mtx = Arc::new(Mutex::new(vk_state));
    let obj_list_mtx = Arc::new(Mutex::new(obj_list));

    let vk_state_mtx_conn = vk_state_mtx.clone();
    let obj_list_mtx_conn = obj_list_mtx.clone();

    thread::spawn(move ||
    {
        println!("Attempt connection !");

        if let Ok(mut conn) = TcpConnection::connect("172.20.10.3:4242")
        {
            let udp = UdpCommunication::new("0.0.0.0:8484").expect("Could not bind UDP");

            println!("Connected !");

            loop
            {
                let size : WindowSize = conn.receive_pod().expect("Could not get dat");
                println!("Got size ! {} {}", size.width, size.height);

                let data = udp.receive();
                println!("Got data !");

                // conn.send_pod(&0u32).expect("Could not send OK signal");

                let Ok((jpg, origin)) = data else { continue; };

                let Ok(data) = turbojpeg::decompress(&jpg, turbojpeg::PixelFormat::RGB) else { println!("Could not decode JPEG"); continue; };

                {
                    let mut vk_state = vk_state_mtx_conn.lock().expect("Could not get Vulkan lock");
                    let mut obj_list = obj_list_mtx_conn.lock().expect("Could not get ObjList lock");

                    let obj = obj_list.get_mut_object(window_id).expect("Could not get window obj");
                    let ObjectKind::Window { window } = &mut obj.kind else { panic!("Window object is not a window quad") };

                    let win_size = window.size();

                    if size.width == win_size.0 && size.height == win_size.1
                    {
                        let buf = Buffer::from_iter(
                            vk_state.allocators.std.clone(),
                            BufferCreateInfo {
                                usage: BufferUsage::TRANSFER_SRC,
                                ..Default::default()
                            },
                            AllocationCreateInfo {
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                            data.pixels.into_iter()
                        ).expect("Could not create buffer");

                        let mut builder = RecordingCommandBuffer::new(vk_state.allocators.cmd.clone(),
                                                                    vk_state.queue_family_index,
                                                                    CommandBufferLevel::Primary,
                                                                    CommandBufferBeginInfo{ usage: CommandBufferUsage::OneTimeSubmit, ..Default::default() }).unwrap();

                        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buf, window.texture.view.image().clone()));
                        builder.end().expect("Could not finalize cmd buffer")
                                .execute(vk_state.queue.clone()).expect("Could not execute cmd buffer")
                                .then_signal_fence_and_flush().unwrap()
                                .wait(None).unwrap();
                    }
                    else
                    {
                        // new_rgb => texture will be sampled and interpolated linearly when magnified or minified
                        // new_pixelated_rgb => texture will be sampled to be rendered like Minecraft
                        let texture = VulkanTexture::new_pixelated_rgb(size.width, size.height, &data.pixels, &vk_state);

                        // Update quad aspect ratio
                        window.refresh_content(texture, &mut obj.obb);
                    }
                }

            }
        }
        else
        {
            panic!("Could not connect to server !");
        }
    });

'main_loop: loop
    {
        while let Some(evt) = xr_state.instance.poll_event(&mut evt_storage).unwrap()
        {
            use xr::Event::*;

            match evt
            {
                SessionStateChanged(e) =>
                {
                    // println!("Entering state: {:?}", e.state());

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

        {
            let mut obj_list = obj_list_mtx.lock().expect("Could not get ObjList lock");

            for gesture in gestures.update(&hand, &obj_list)
            {
                // println!("Gesture: {:?}", gesture);

                let obj = obj_list.get_mut_object(gesture.id).unwrap();

                match &mut obj.kind
                {
                    ObjectKind::DebugCube | ObjectKind::TintedCube { tint: _ } | ObjectKind::TexturedQuad { texture: _ } =>
                    {
                        if gesture.phase != GesturePhase::Cancelled
                        {
                            match gesture.kind
                            {
                                GestureKind::Grab { tips: _} | GestureKind::Ray =>
                                {
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
                    },
                    ObjectKind::Window { window } =>
                    {
                        let Some(hand) = hand else { break };

                        if let Some(event) = window.handle_gesture(&mut obj.transform, &hand, &gesture, &gestures)
                        {
                            match event
                            {
                                WindowEvent::ContentInteract { tips } =>
                                {
                                    for tip in tips
                                    {
                                        colliding_tips[tip.to_idx()] = true;
                                    }
                                },
                                WindowEvent::Close =>
                                {
                                    obj.kind = ObjectKind::Empty;
                                },
                                WindowEvent::Drag => {}
                            }
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

                {
                    let hand_obj = obj_list.get_mut_object(hand_id).expect("Could not get hand obj");

                    // yes this builds, even if rust-analyzer says otherwise
                    let ObjectKind::Hand { hand: obj_hand, buffer } = &mut hand_obj.kind else { panic!("Hand object is not hand") };

                    *obj_hand = hand.unwrap_or(Hand::IDENTITY);
                }

                {
                    let raycast_obj = obj_list.get_mut_object(raycast_id).expect("Could not get raycast obj");

                    // yes this builds, even if rust-analyzer says otherwise
                    let ObjectKind::Raycast { ray, buffer } = &mut raycast_obj.kind else { panic!("Hand object is not hand") };

                    *ray = hand.and_then(|hand| Some(hand.compute_ray())).unwrap_or(Ray::X);
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

        {
            let mut vk_state = vk_state_mtx.lock().expect("Could not get Vulkan lock");

            let mut swapchain = swapchain.get_or_insert_with(|| VulkanSwapchain::new(&xr_state, &vk_state));

            swapchain.next_frame();

            // Writing Command Buffer

            let mut builder = RecordingCommandBuffer::new(vk_state.allocators.cmd.clone(),
                                                        vk_state.queue_family_index,
                                                        CommandBufferLevel::Primary,
                                                        CommandBufferBeginInfo{ usage: CommandBufferUsage::OneTimeSubmit, ..Default::default() }).unwrap();

            builder.begin_render_pass(
                    RenderPassBeginInfo
                    {
                        clear_values: vec![Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])), Some(ClearValue::Depth(1.0))],
                        ..RenderPassBeginInfo::framebuffer(swapchain.get_framebuffer().clone())
                    },
                    SubpassBeginInfo
                    {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    }
                ).unwrap()
                .set_viewport(0, [Viewport { offset: [0.0, 0.0], extent: [width as f32, height as f32], depth_range: 0.0..=1.0 }].into_iter().collect()).unwrap()
                .set_scissor(0, [Scissor { offset: [0, 0], extent: [width, height] }].into_iter().collect()).unwrap();

            let mut render_state = RenderState
            {
                swapchain: &swapchain,
                builder: &mut builder
            };

            let mut obj_list = obj_list_mtx.lock().expect("Could not get ObjList lock");

            unsafe
            {
    'top:       for (_, obj) in obj_list.iter()
                {
                    match &obj.kind
                    {
                        ObjectKind::Empty => {},
                        ObjectKind::DebugCube =>
                        {
                            vk_state.render_debug_cube(&obj.transform, &mut render_state);
                        },
                        ObjectKind::TintedCube { tint } =>
                        {
                            vk_state.render_tinted_cube(&obj.transform, tint, &mut render_state);
                        },
                        ObjectKind::TexturedQuad { texture } =>
                        {
                            vk_state.render_textured_quad(&obj.transform, texture, &mut render_state);
                        },
                        ObjectKind::Hand { hand, buffer } =>
                        {
                            if !show_debug { continue 'top; }

                            let tint = glam::vec4(0.0, 0.0, 1.0, 1.0);

                            vk_state.buffers.update_hand_wireframe_buffer(buffer, hand);
                            vk_state.render_wireframe(buffer.as_ref(), &tint, &mut render_state);
                        },
                        ObjectKind::Raycast { ray, buffer } =>
                        {
                            if !show_debug { continue 'top; }

                            let tint = get_raycast_tint(ray, &cube_ids, &obj_list);

                            vk_state.buffers.update_raycast_buffer(buffer, ray);
                            vk_state.render_wireframe(buffer.as_ref(), &tint, &mut render_state);
                        },
                        ObjectKind::Window { window } =>
                        {
                            window.draw(&obj.transform, &vk_state, &mut render_state);

                            let obb_transform = Transform::new(
                                obj.transform.pos,
                                obj.transform.rot,
                                obj.transform.size * obj.obb.and_then(|obb| Some(obb.size)).unwrap_or(Vec3::ZERO)
                            );

                            // vk_state.render_tinted_cube_wireframe(&obb_transform, &vec4(0.0, 0.0, 1.0, 0.66), &mut render_state);
                        }
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

            swapchain.update_global_unform(&GlobalUniformData
            {
                left: view_to_matrix(&views[0]),
                right: view_to_matrix(&views[1])
            });

            swapchain.wait_frame();

            let _ = cmd_buffer.execute(vk_state.queue.clone()).unwrap()
                .then_signal_fence_and_flush().unwrap()
                .wait(None).unwrap();

            swapchain.release_frame();

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
        }

        last_frame = Instant::now();

        // println!("Frame displayed");
    }
}

fn get_raycast_tint(ray: &Ray, cube_ids: &[ObjectID], obj_list: &ObjectList) -> glam::Vec4
{
    let does_intersects = 'block: {

        for &cube_id in cube_ids
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

    return if does_intersects { glam::vec4(0.0, 1.0, 0.0, 0.0) } else { glam::vec4(1.0, 0.0, 0.0, 0.0) };
}
