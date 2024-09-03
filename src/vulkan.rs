use std::{default, ffi::CStr, marker::PhantomData, mem::transmute, sync::Arc};

use buffer::BufferContents;
use image::ImageAspects;
use pipeline::{graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{DepthState, DepthStencilState, DepthStencilStateFlags}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::{CullMode, FrontFace, RasterizationState}, vertex_input::{Vertex, VertexDefinition, VertexInputState}, viewport::ViewportState, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::{*, library::*, instance::*, device::*, device::physical::*, render_pass::*};
use ash::vk::{self, Handle};

use crate::openxr::XRSetupState;

// #[repr(C)]
// #[derive(BufferContents)]
// pub struct GlobalUniformData
// {
//     #[format(R32G32B32_SFLOAT)]
//     left_x_col: glam::Vec4,
//     #[format(R32G32B32_SFLOAT)]
//     left_y_col: glam::Vec4,
//     #[format(R32G32B32_SFLOAT)]
//     left_z_col: glam::Vec4,
//     #[format(R32G32B32_SFLOAT)]
//     left_w_col: glam::Vec4,

//     #[format(R32G32B32_SFLOAT)]
//     right_x_col: glam::Vec4,
//     #[format(R32G32B32_SFLOAT)]
//     right_y_col: glam::Vec4,
//     #[format(R32G32B32_SFLOAT)]
//     right_z_col: glam::Vec4,
//     #[format(R32G32B32_SFLOAT)]
//     right_w_col: glam::Vec4,
// }


#[repr(C)]
#[derive(BufferContents)]
pub struct GlobalUniformData
{
    pub left: glam::Mat4,
    pub right: glam::Mat4
}

// #[repr(C)]
// #[derive(BufferContents)]
// pub struct ObjectUniformData
// {
//     #[format(R32G32B32A32_SFLOAT)]
//     transform_x_col: glam::Vec4,
//     #[format(R32G32B32A32_SFLOAT)]
//     transform_y_col: glam::Vec4,
//     #[format(R32G32B32A32_SFLOAT)]
//     transform_z_col: glam::Vec4,
//     #[format(R32G32B32A32_SFLOAT)]
//     transform_w_col: glam::Vec4,

//     #[format(R32G32B32_SFLOAT)]
//     tint: glam::Vec3
// }

#[repr(C)]
#[derive(BufferContents)]
pub struct ObjectData
{
    pub transform: glam::Mat4,
    pub tint: glam::Vec4
}


#[repr(C)]
#[derive(BufferContents, Vertex)]
pub struct BaseVertex
{
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3]
}

pub struct VulkanState
{
    pub instance: Arc<Instance>,
    pub pdevice: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub queue_family_index: u32,
    pub render_pass: Arc<RenderPass>,
    pub pipeline: Arc<GraphicsPipeline>,
}

impl VulkanState
{
    // pub fn init() -> Option<Self>
    // {
    //     let library = VulkanLibrary::new().ok()?;

    //     let instance = Instance::new(library, InstanceCreateInfo
    //     {
    //         application_name: Some("openxr_tests".to_string()),
    //         application_version: Version::major_minor(1, 0),
    //         ..Default::default()
    //     }).ok()?;

    //     let (pdevice, queue_family_index) = instance.enumerate_physical_devices().ok()?
    //         .filter_map(|pdevice|
    //         {
    //             pdevice.queue_family_properties().iter().enumerate()
    //                 .position(|(_i, q)|
    //                 {
    //                     q.queue_flags.contains(QueueFlags::GRAPHICS)
    //                 })
    //                 .map(|i| (pdevice, i as u32))
    //         })
    //         .min_by_key(|(pdevice, _)|
    //         {
    //             match pdevice.properties().device_type
    //             {
    //                 PhysicalDeviceType::DiscreteGpu => 0,
    //                 PhysicalDeviceType::IntegratedGpu => 1,
    //                 PhysicalDeviceType::VirtualGpu => 2,
    //                 PhysicalDeviceType::Cpu => 3,
    //                 _ => 4,
    //             }
    //         })?;

    //     let (device, mut queues) = Device::new(pdevice.clone(), DeviceCreateInfo
    //     {
    //         enabled_features: Features
    //         {
    //             ..Default::default()
    //         },
    //         queue_create_infos: vec![
    //             QueueCreateInfo
    //             {
    //                 queue_family_index,
    //                 ..Default::default()
    //             }
    //         ],
    //         ..Default::default()
    //     }).ok()?;

    //     let queue = queues.next()?;

    //     Some(VulkanState
    //     {
    //         instance,
    //         pdevice,
    //         device,
    //         queue,
    //         queue_family_index
    //     })
    // }

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
                .position(|(i, info)|
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

            // mod vs
            // {
            //     vulkano_shaders::shader! {
            //         ty: "vertex",
            //         src: r"
            //             #version 450
            //             #extension GL_EXT_multiview : require

            //             layout(push_constant) uniform data { mat4 views[2]; } constants;

            //             layout(location = 0) in vec3 position;
            //             layout(location = 1) in vec3 color;

            //             layout(location = 0) out vec3 outColor;

            //             void main() {
            //                 gl_Position = constants.views[gl_ViewIndex] * (vec4(position, 1) + vec4(0, 0, 3, 0));
            //                 outColor = color;
            //             }
            //         "
            //     }
            // }

            mod vs
            {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 450
                        #extension GL_EXT_multiview : require

                        layout(set = 0, binding = 0) uniform GlobalData
                        {
                            mat4 proj[2];
                        } global;

                        layout(push_constant) uniform ObjectData
                        {
                            mat4 transform;
                            vec4 tint;
                        } object;

                        layout(location = 0) in vec3 position;
                        layout(location = 1) in vec4 color;

                        layout(location = 0) out vec4 outColor;

                        void main()
                        {
                            gl_Position = global.proj[gl_ViewIndex] * object.transform * vec4(position, 1);
                            outColor = color * object.tint;
                        }
                    "
                }
            }

            mod fs
            {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 450

                        layout(location = 0) in vec4 inColor;
                        layout(location = 0) out vec4 outColor;

                        void main()
                        {
                            outColor = inColor;
                        }
                    "
                }
            }

            println!("Creating pipeline");

            let pipeline =
            {
                let vs = vs::load(vulkano_device.clone()).unwrap()
                    .entry_point("main").unwrap();

                let fs = fs::load(vulkano_device.clone()).unwrap()
                    .entry_point("main").unwrap();

                let vertex_input_state = BaseVertex::per_vertex().definition(&vs).unwrap();

                let stages =
                [
                    PipelineShaderStageCreateInfo::new(vs),
                    PipelineShaderStageCreateInfo::new(fs),
                ];

                let layout = PipelineLayout::new(
                    vulkano_device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                        .into_pipeline_layout_create_info(vulkano_device.clone())
                        .unwrap()
                ).ok()?;

                let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

                GraphicsPipeline::new(vulkano_device.clone(), None,
                    GraphicsPipelineCreateInfo
                    {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(ViewportState::default()),
                        rasterization_state: Some(RasterizationState
                        {
                            front_face: FrontFace::CounterClockwise,
                            cull_mode: CullMode::Front,
                            ..Default::default()
                        }),
                        multisample_state: Some(MultisampleState::default()),
                        color_blend_state: Some(ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments() as u32,
                            ColorBlendAttachmentState::default())
                        ),
                        dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                        subpass: Some(subpass.into()),
                        depth_stencil_state: Some(DepthStencilState
                        {
                            depth: Some(DepthState::simple()),
                            depth_bounds: None,
                            stencil: None,
                            ..Default::default()
                        }),
                        ..GraphicsPipelineCreateInfo::layout(layout)
                    }
                ).unwrap()
            };

            println!("Vulkan setup");

            Some(VulkanState
            {
                instance: vulkano_instance,
                pdevice: vulkano_pdevice,
                device: vulkano_device,
                queue,
                queue_family_index,
                render_pass,
                pipeline
            })
        }
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
