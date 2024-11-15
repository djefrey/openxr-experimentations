use core::f32;

use glam::*;

use crate::{ray::Ray, Transform};

#[derive(Debug, Clone, Copy)]
pub struct OBB
{
    pub size: Vec3
}

#[derive(Debug, Clone, Copy)]
pub struct ComputedOOB
{
    center: Vec3,
    rot: Quat,
    half_size: Vec3,
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
    pub const CUBE_OBB : OBB = OBB { size: Vec3::ONE };

    pub fn new(size: Vec3) -> Self
    {
        Self
        {
            size
        }
    }

    pub fn set_size(&mut self, size: Vec3) -> &mut Self
    {
        self.size = size;

        self
    }

    pub fn compute_obb(&self, transform: &Transform) -> ComputedOOB
    {
        let x_normal = transform.rot.mul_vec3(Vec3::X);
        let y_normal = transform.rot.mul_vec3(Vec3::Y);
        let z_normal = transform.rot.mul_vec3(Vec3::Z);

        let half = self.size * transform.size / 2.0;

        let vertices : [Vec3; 8] =
        [
            transform.pos + x_normal * half.x + y_normal * half.y + z_normal * half.z,
            transform.pos - x_normal * half.x + y_normal * half.y + z_normal * half.z,
            transform.pos + x_normal * half.x - y_normal * half.y + z_normal * half.z,
            transform.pos + x_normal * half.x + y_normal * half.y - z_normal * half.z,
            transform.pos - x_normal * half.x - y_normal * half.y + z_normal * half.z,
            transform.pos - x_normal * half.x + y_normal * half.y - z_normal * half.z,
            transform.pos + x_normal * half.x - y_normal * half.y - z_normal * half.z,
            transform.pos - x_normal * half.x - y_normal * half.y - z_normal * half.z,
        ];

        ComputedOOB
        {
            center: transform.pos,
            rot: transform.rot,
            half_size: half,
            vertices
        }
    }
}

impl ComputedOOB
{
    pub fn does_intersects_ray(&self, ray: &Ray) -> Option<f32>
    {
        let local_origin = self.rot.inverse() * (self.center - ray.origin);
        let local_dir = self.rot.inverse() * -ray.dir;

        let axis = [
            (
                (-self.half_size.x - local_origin.x) / local_dir.x,
                ( self.half_size.x - local_origin.x) / local_dir.x
            ),
            (
                (-self.half_size.y - local_origin.y) / local_dir.y,
                ( self.half_size.y - local_origin.y) / local_dir.y
            ),
            (
                (-self.half_size.z - local_origin.z) / local_dir.z,
                ( self.half_size.z - local_origin.z) / local_dir.z
            )
        ];

        let mut t_enter = f32::MIN;
        let mut t_exit = f32::MAX;

        for (t1, t2) in axis
        {
            t_enter = t_enter.max(t1.min(t2));
            t_exit  = t_exit.min(t1.max(t2));
        }

        return if t_enter <= t_exit && t_exit >= 0.0 { Some(t_enter) } else { None }
    }

    pub fn does_intersects_obb(&self, rhs: &ComputedOOB) -> bool
    {
        let lhs_x_normal = self.rot.mul_vec3(Vec3::X);
        let lhs_y_normal = self.rot.mul_vec3(Vec3::Y);
        let lhs_z_normal = self.rot.mul_vec3(Vec3::Z);

        let rhs_x_normal = rhs.rot.mul_vec3(Vec3::X);
        let rhs_y_normal = rhs.rot.mul_vec3(Vec3::Y);
        let rhs_z_normal = rhs.rot.mul_vec3(Vec3::Z);

        let axes : [Vec3; 15] =
        [
            lhs_x_normal,
            lhs_y_normal,
            lhs_z_normal,

            rhs_x_normal,
            rhs_y_normal,
            rhs_z_normal,

            lhs_x_normal.cross(rhs_x_normal),
            lhs_x_normal.cross(rhs_y_normal),
            lhs_x_normal.cross(rhs_z_normal),

            lhs_y_normal.cross(rhs_x_normal),
            lhs_y_normal.cross(rhs_y_normal),
            lhs_y_normal.cross(rhs_z_normal),

            lhs_z_normal.cross(rhs_x_normal),
            lhs_z_normal.cross(rhs_y_normal),
            lhs_z_normal.cross(rhs_z_normal),
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
