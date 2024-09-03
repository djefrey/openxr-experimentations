use core::f32;
use std::ops::RangeInclusive;

use glam::*;

use crate::Transform;

pub struct OBB
{
    pub center: Vec3,
    pub size: Vec3,
    pub rot: Quat
}

struct ComputedOOB
{
    x_normal: Vec3,
    y_normal: Vec3,
    z_normal: Vec3,
    vertices: [Vec3; 8]
}

#[derive(Debug)]
struct CollisionInterval
{
    min: f32,
    max: f32
}

impl OBB
{
    pub fn new(center: Vec3, size: Vec3, rot: Quat) -> Self
    {
        Self
        {
            center,
            size,
            rot
        }
    }

    pub fn from_transform(transform: &Transform) -> Self
    {
        Self
        {
            center: transform.pos,
            size: transform.size,
            rot: transform.rot
        }
    }

    pub fn set_center(&mut self, center: Vec3) -> &mut Self
    {
        self.center = center;

        self
    }

    pub fn set_size(&mut self, size: Vec3) -> &mut Self
    {
        self.size = size;

        self
    }

    pub fn set_rot(&mut self, rot: Quat) -> &mut Self
    {
        self.rot = rot;

        self
    }

    pub fn does_collide(&self, other: &OBB) -> bool
    {
        let rpos = other.center - self.center;
        let self_computed = self.compute_obb();
        let other_computed = other.compute_obb();

        return self_computed.intersects(&other_computed);
    }

    fn compute_obb(&self) -> ComputedOOB
    {
        let x_normal = self.rot.mul_vec3(Vec3::X);
        let y_normal = self.rot.mul_vec3(Vec3::Y);
        let z_normal = self.rot.mul_vec3(Vec3::Z);

        let half = self.size / 2.0;

        let vertices : [Vec3; 8] =
        [
            self.center + x_normal * half.x + y_normal * half.y + z_normal * half.z,
            self.center - x_normal * half.x + y_normal * half.y + z_normal * half.z,
            self.center + x_normal * half.x - y_normal * half.y + z_normal * half.z,
            self.center + x_normal * half.x + y_normal * half.y - z_normal * half.z,
            self.center - x_normal * half.x - y_normal * half.y + z_normal * half.z,
            self.center - x_normal * half.x + y_normal * half.y - z_normal * half.z,
            self.center + x_normal * half.x - y_normal * half.y - z_normal * half.z,
            self.center - x_normal * half.x - y_normal * half.y - z_normal * half.z,
        ];

        ComputedOOB
        {
            x_normal,
            y_normal,
            z_normal,
            vertices
        }
    }
}

impl ComputedOOB
{
    fn intersects(&self, rhs: &ComputedOOB) -> bool
    {
        let axes : [Vec3; 15] =
        [
            self.x_normal,
            self.y_normal,
            self.z_normal,

            rhs.x_normal,
            rhs.y_normal,
            rhs.z_normal,

            self.x_normal.cross(rhs.x_normal),
            self.x_normal.cross(rhs.y_normal),
            self.x_normal.cross(rhs.z_normal),

            self.y_normal.cross(rhs.x_normal),
            self.y_normal.cross(rhs.y_normal),
            self.y_normal.cross(rhs.z_normal),

            self.z_normal.cross(rhs.x_normal),
            self.z_normal.cross(rhs.y_normal),
            self.z_normal.cross(rhs.z_normal),
        ];

        for axis in axes.into_iter()
        {
            if !self.compute_axis_collision(&rhs, axis)
            {
                return false;
            }
        }

        return true;
    }

    fn compute_axis_collision(&self, rhs: &ComputedOOB, axis: Vec3) -> bool
    {
        let a = self.compute_interval_for_axis(axis);
        let b = rhs.compute_interval_for_axis(axis);

        a.min <= b.max && b.min <= a.max
    }

    fn compute_interval_for_axis(&self, axis: Vec3) -> CollisionInterval
    {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for &vertex in self.vertices.iter()
        {
            let dot = axis.dot(vertex);
            min = min.min(dot);
            max = max.max(dot);
        }

        CollisionInterval { min, max }
    }
}
