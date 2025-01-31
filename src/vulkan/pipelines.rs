use std::sync::Arc;

use vulkano::{device::Device, pipeline::{graphics::{color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents}, depth_stencil::{DepthState, DepthStencilState}, input_assembly::{InputAssemblyState, PrimitiveTopology}, multisample::MultisampleState, rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState}, vertex_input::{Vertex, VertexDefinition, VertexInputState}, viewport::ViewportState, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo}, render_pass::{RenderPass, Subpass}, shader::ShaderStages};

use crate::vulkan::buffers::{LineVertex, TexturedVertex, TintedVertex};

pub struct VulkanPipelines
{
    pub tinted: Arc<GraphicsPipeline>,
    pub tinted_debug: Arc<GraphicsPipeline>,
    pub textured: Arc<GraphicsPipeline>,
    pub line: Arc<GraphicsPipeline>,
    pub cursor: Arc<GraphicsPipeline>,
}

impl VulkanPipelines
{
    pub fn new(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Self
    {
        mod tinted_vs
        {
            vulkano_shaders::shader! { ty: "vertex", path: "./shaders/tinted.vert" }
        }

        mod textured_vs
        {
            vulkano_shaders::shader! { ty: "vertex", path: "./shaders/textured.vert" }
        }

        mod line_vs
        {
            vulkano_shaders::shader! { ty: "vertex", path: "./shaders/line.vert" }
        }

        mod fullscreen_fs
        {
            vulkano_shaders::shader! { ty: "vertex", path: "./shaders/fullscreen.vert" }
        }

        mod tinted_fs
        {
            vulkano_shaders::shader! { ty: "fragment", path: "./shaders/tinted.frag" }
        }

        mod textured_fs
        {
            vulkano_shaders::shader! { ty: "fragment", path: "./shaders/textured.frag" }
        }

        mod cursor_fs
        {
            vulkano_shaders::shader! { ty: "fragment", path: "./shaders/cursor.frag" }
        }

        let tinted =
        {
            let vs = tinted_vs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = tinted_fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let vertex_input_state = TintedVertex::per_vertex().definition(&vs).unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = {
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

                create_info.set_layouts[0].bindings.get_mut(&0).unwrap().stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

                PipelineLayout::new(
                    device.clone(),
                    create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap()
                ).ok().unwrap()
            };

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(device.clone(), None,
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
                        ColorBlendAttachmentState
                        {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_enable: true,
                            color_write_mask: ColorComponents::all()
                        })
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

        let tinted_debug =
        {
            let vs = tinted_vs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = tinted_fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let vertex_input_state = TintedVertex::per_vertex().definition(&vs).unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = {
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

                create_info.set_layouts[0].bindings.get_mut(&0).unwrap().stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

                PipelineLayout::new(
                    device.clone(),
                    create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap()
                ).ok().unwrap()
            };

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(device.clone(), None,
                GraphicsPipelineCreateInfo
                {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState
                    {
                        polygon_mode: PolygonMode::Line,
                        front_face: FrontFace::CounterClockwise,
                        cull_mode: CullMode::Front,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments() as u32,
                        ColorBlendAttachmentState
                        {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_enable: true,
                            color_write_mask: ColorComponents::all()
                        })
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

        let textured =
        {
            let vs = textured_vs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = textured_fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let vertex_input_state = TexturedVertex::per_vertex().definition(&vs).unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = {
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

                create_info.set_layouts[0].bindings.get_mut(&0).unwrap().stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

                PipelineLayout::new(
                    device.clone(),
                    create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap()
                ).ok().unwrap()
            };

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(device.clone(), None,
                GraphicsPipelineCreateInfo
                {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState
                    {
                        front_face: FrontFace::CounterClockwise,
                        cull_mode: CullMode::None, // Cull disabled to see both faces
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments() as u32,
                        ColorBlendAttachmentState
                        {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_enable: true,
                            color_write_mask: ColorComponents::all()
                        })
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

        let line =
        {
            let vs = line_vs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = tinted_fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let vertex_input_state = LineVertex::per_vertex().definition(&vs).unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = {
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

                create_info.set_layouts[0].bindings.get_mut(&0).unwrap().stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

                PipelineLayout::new(
                    device.clone(),
                    create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap()
                ).ok().unwrap()
            };

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(device.clone(), None,
                GraphicsPipelineCreateInfo
                {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState
                    {
                        topology: PrimitiveTopology::LineList,
                        ..Default::default()
                    }),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments() as u32,
                        ColorBlendAttachmentState
                        {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_enable: true,
                            color_write_mask: ColorComponents::all()
                        })
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

        let cursor =
        {
            let vs = fullscreen_fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = cursor_fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = {
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

                println!("{:?}", create_info.set_layouts);

                create_info.set_layouts[0].bindings.get_mut(&0).unwrap().stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

                PipelineLayout::new(
                    device.clone(),
                    create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap()
                ).ok().unwrap()
            };

            let subpass = Subpass::from(render_pass.clone(), 1).unwrap();

            GraphicsPipeline::new(device.clone(), None,
                GraphicsPipelineCreateInfo
                {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(VertexInputState::new()),
                    input_assembly_state: Some(InputAssemblyState
                    {
                        topology: PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    }),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments() as u32,
                        ColorBlendAttachmentState
                        {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_enable: true,
                            color_write_mask: ColorComponents::all()
                        })
                    ),
                    dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    depth_stencil_state: None,
                    ..GraphicsPipelineCreateInfo::layout(layout)
                }
            ).unwrap()
        };

        Self
        {
            tinted,
            tinted_debug,
            textured,
            line,
            cursor,
        }
    }
}
