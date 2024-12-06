use std::sync::Arc;

use crate::{gestures::Hand, ray::Ray};

use glam::{Mat4, Vec4};
use openxr as xr;
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}};
use vulkano_macros::{BufferContents, Vertex};

use super::VulkanAllocators;

const TRIANGLE_VERTICES : [TintedVertex; 3] = [
    TintedVertex { position: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0] },
    TintedVertex { position: [1.0, 0.0, 0.0], color: [0.0, 1.0, 0.0] },
    TintedVertex { position: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
];

const DEBUG_CUBE : [TintedVertex; 8] = [
    // BOT
    TintedVertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 0.0] },
    TintedVertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0] },
    TintedVertex { position: [-0.5, -0.5,  0.5], color: [0.0, 0.0, 1.0] },
    TintedVertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 0.0] },

    // TOP
    TintedVertex { position: [-0.5,  0.5, -0.5], color: [1.0, 0.0, 1.0] },
    TintedVertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 1.0] },
    TintedVertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 0.0] },
];

const WHITE_CUBE : [TintedVertex; 8] = [
    // BOT
    TintedVertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [-0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 1.0, 1.0] },

    // TOP
    TintedVertex { position: [-0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [ 0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0] },
    TintedVertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 1.0, 1.0] },
];

const QUAD : [TexturedVertex; 4] = [
    TexturedVertex { position: [-0.5, -0.5, 0.0], uv: [0.0, 1.0] }, // BOT LEFT
    TexturedVertex { position: [-0.5,  0.5, 0.0], uv: [0.0, 0.0] }, // TOP LEFT
    TexturedVertex { position: [ 0.5, -0.5, 0.0], uv: [1.0, 1.0] }, // BOT RIGHT
    TexturedVertex { position: [ 0.5,  0.5, 0.0], uv: [1.0, 0.0] }, // TOP RIGHT
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

const QUAD_INDICES : [u16; 6] = [
    0, 1, 2,
    2, 1, 3
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

#[repr(C)]
#[derive(Debug, Clone, Copy, BufferContents)]
pub struct TintedObjectData
{
    pub transform: Mat4,
    pub tint: Vec4
}


#[repr(C)]
#[derive(Debug, Clone, Copy, Vertex, bytemuck::Pod, bytemuck::Zeroable)] // BufferContents impl by Pod + Zeroable
pub struct TintedVertex
{
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Vertex, bytemuck::Pod, bytemuck::Zeroable)] // BufferContents impl by Pod + Zeroable
pub struct TexturedVertex
{
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Vertex, bytemuck::Pod, bytemuck::Zeroable)] // BufferContents impl by Pod + Zeroable
pub struct LineVertex
{
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct HandBuffer(pub Subbuffer<[LineVertex; 64]>);

#[derive(Debug, Clone)]
pub struct RaycastBuffer(pub Subbuffer<[LineVertex; 2]>);

pub struct VulkanBuffers
{
    std_alloc: Arc<StandardMemoryAllocator>,
    pub debug_cube_vertex: Subbuffer<[TintedVertex]>,
    pub cube_vertex: Subbuffer<[TintedVertex]>,
    pub cube_index: Subbuffer<[u16]>,
    pub quad_vertex: Subbuffer<[TexturedVertex]>,
    pub quad_index: Subbuffer<[u16]>,
}

impl VulkanBuffers
{
    pub fn new(allocators: &VulkanAllocators) -> Self
    {
        let std = &allocators.std;

        let debug_cube_vertex_buffer = Buffer::from_iter(std.clone(),
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

        let cube_vertex_buffer = Buffer::from_iter(std.clone(),
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

        let cube_index_buffer = Buffer::from_iter(std.clone(),
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

        let quad_vertex_buffer = Buffer::from_iter(std.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            QUAD.into_iter())
        .unwrap();

        let quad_index_buffer = Buffer::from_iter(std.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            QUAD_INDICES.into_iter())
        .unwrap();

        Self
        {
            std_alloc: std.clone(),
            debug_cube_vertex: debug_cube_vertex_buffer,
            cube_vertex: cube_vertex_buffer,
            cube_index: cube_index_buffer,
            quad_vertex: quad_vertex_buffer,
            quad_index: quad_index_buffer
        }
    }
}

// HAND BUFFER
impl VulkanBuffers
{
    pub fn new_hand_wireframe_buffer(&self) -> HandBuffer
    {
        let buffer = Buffer::from_iter(self.std_alloc.clone(),
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

        HandBuffer(buffer.reinterpret())
    }

    pub unsafe fn update_hand_wireframe_buffer(&self, buffer: &HandBuffer, hand: &Hand)
    {
        (*buffer.0.write().unwrap()).copy_from_slice(&HAND_LINES.map(|i|
        {
            LineVertex
            {
                position: hand[i].pos.to_array()
            }
        }));
    }
}

// RAYCAST BUFFER
impl VulkanBuffers
{
    pub fn new_raycast_buffer(&self) -> RaycastBuffer
    {
        let buffer = Buffer::from_iter(self.std_alloc.clone(),
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

        RaycastBuffer(buffer.reinterpret())
    }

    pub unsafe fn update_raycast_buffer(&self, buffer: &RaycastBuffer, ray: &Ray)
    {
        (*buffer.0.write().unwrap()).copy_from_slice(&ray.to_points(5.0).map(|p|
        {
            LineVertex
            {
                position: p.to_array()
            }
        }));
    }
}

impl AsRef<Subbuffer<[LineVertex]>> for HandBuffer
{
    fn as_ref(&self) -> &Subbuffer<[LineVertex]>
    {
        self.0.reinterpret_ref()
    }
}

impl AsRef<Subbuffer<[LineVertex]>> for RaycastBuffer
{
    fn as_ref(&self) -> &Subbuffer<[LineVertex]>
    {
        self.0.reinterpret_ref()
    }
}
