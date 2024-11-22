use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Ray
{
    pub origin : Vec3,
    pub dir : Vec3
}

impl Ray
{
    pub const X: Ray = Ray { origin: Vec3::ZERO, dir: Vec3::X };

    pub fn new(origin: Vec3, dir: Vec3) -> Self
    {
        Self
        {
            origin,
            dir: dir.normalize()
        }
    }

    pub fn new_assume_normalize(origin: Vec3, dir: Vec3) -> Self
    {
        Self
        {
            origin,
            dir
        }
    }

    pub fn from_points(a: Vec3, b: Vec3) -> Self
    {
        let dir = (b - a).normalize();

        return Ray::new(a, dir);
    }

    pub fn to_points(&self, distance: f32) -> [Vec3; 2]
    {
        return [self.origin, self.origin + self.dir * distance];
    }
}
