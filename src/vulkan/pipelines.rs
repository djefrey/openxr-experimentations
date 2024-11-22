use std::sync::Arc;

use vulkano::{device::Device, pipeline::{graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{DepthState, DepthStencilState}, input_assembly::{InputAssemblyState, PrimitiveTopology}, multisample::MultisampleState, rasterization::{CullMode, FrontFace, RasterizationState}, vertex_input::{Vertex, VertexDefinition}, viewport::ViewportState, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo}, render_pass::{RenderPass, Subpass}};

use crate::vulkan::buffers::{LineVertex, TintedVertex};

pub struct VulkanPipelines
{
    pub tinted: Arc<GraphicsPipeline>,
    pub line: Arc<GraphicsPipeline>,
}

impl VulkanPipelines
{
    pub fn new(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Self
    {
        mod vs
        {
            vulkano_shaders::shader! { ty: "vertex", path: "./shaders/tinted.vert" }
        }

        mod line_vs
        {
            vulkano_shaders::shader! { ty: "vertex", path: "./shaders/line.vert" }
        }

        mod fs
        {
            vulkano_shaders::shader! { ty: "fragment", path: "./shaders/tinted.frag" }
        }

        let tinted =
        {
            let vs = vs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let vertex_input_state = TintedVertex::per_vertex().definition(&vs).unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap()
            ).ok().unwrap();

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


        let line =
        {
            let vs = line_vs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let fs = fs::load(device.clone()).unwrap()
                .entry_point("main").unwrap();

            let vertex_input_state = LineVertex::per_vertex().definition(&vs).unwrap();

            let stages =
            [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap()
            ).ok().unwrap();

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

        Self
        {
            tinted,
            line
        }
    }
}
