use glam::Vec4;

use crate::{gestures::Hand, obb::{ComputedOOB, OBB}, ray::Ray, vulkan::buffers::{HandBuffer, RaycastBuffer}, Transform};

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd)]
pub struct ObjectID(usize);

#[derive(Debug, Clone)]
pub enum ObjectKind
{
    DebugCube,
    TintedCube { tint: Vec4 },
    Hand { hand: Hand, buffer: HandBuffer },
    Raycast { ray: Ray, buffer: RaycastBuffer }
}

pub struct Object
{
    pub kind : ObjectKind,
    pub transform: Transform,
    pub obb: Option<OBB>
}

impl Object
{
    pub fn compute_obb(&self) -> Option<ComputedOOB>
    {
        Some(self.obb?.compute_obb(&self.transform))
    }
}

pub struct ObjectList
{
    pub list: Vec<Object>
}

impl ObjectList
{
    pub fn new() -> Self
    {
        Self
        {
            list: Vec::new()
        }
    }

    pub fn new_object(&mut self, kind: ObjectKind, transform: Transform, obb: Option<OBB>) -> ObjectID
    {
        let id = ObjectID(self.list.len());
        let obj = Object
        {
            kind,
            transform,
            obb
        };

        self.list.push(obj);

        return id;
    }

    pub fn get_object(&self, id: ObjectID) -> Option<&Object>
    {
        return self.list.get(id.0);
    }

    pub fn get_mut_object(&mut self, id: ObjectID) -> Option<&mut Object>
    {
        return self.list.get_mut(id.0);
    }
}

pub struct ObjectListIterator<'a>
{
    list: &'a ObjectList,
    index: usize
}

impl<'a> Iterator for ObjectListIterator<'a>
{
    type Item = (ObjectID, &'a Object);

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.index < self.list.list.len()
        {
            let res = Some((ObjectID(self.index), &self.list.list[self.index]));
            self.index += 1;

            return res;
        }
        else
        {
            return None;
        }
    }
}

impl<'a> ObjectList
{
    pub fn iter(&'a self) -> ObjectListIterator<'a>
    {
        return ObjectListIterator { list: self, index: 0 };
    }
}
