use std::ops::Index;

use glam::Vec3;

use openxr::{self as xr};

use crate::{obb::OBB, object::{Object, ObjectID, ObjectList}, openxr::XRState, ray::Ray, Transform};

pub enum Gesture
{
    Tap { id: ObjectID },
    Drag { id: ObjectID }
}

pub struct GestureState
{
    previous_wrist: Option<Transform>,
    current_wrist: Option<Transform>,
}

pub struct Hand([Transform; 26]);

pub struct HandTip(usize);
impl HandTip
{
    pub const THUMB: HandTip = Self(0);
    pub const INDEX: HandTip = Self(1);
    pub const MIDDLE: HandTip = Self(2);
    pub const RING: HandTip = Self(3);
    pub const LITTLE: HandTip = Self(4);

    pub fn to_joint(&self) -> xr::HandJointEXT
    {
        match self.0
        {
            0 => xr::HandJointEXT::THUMB_TIP,
            1 => xr::HandJointEXT::INDEX_TIP,
            2 => xr::HandJointEXT::MIDDLE_TIP,
            3 => xr::HandJointEXT::RING_TIP,
            4 => xr::HandJointEXT::LITTLE_TIP,
            _ => panic!("Invalid tip")
        }
    }

    pub fn to_idx(&self) -> usize
    {
        return self.0;
    }
}

impl Hand
{
    pub const IDENTITY : Hand = Hand([Transform::IDENTITY; 26]);

    pub fn from_xr_state(xr: &XRState, predicted_time: xr::Time) -> Option<Hand>
    {
        let res = xr.stage.locate_hand_joints(&xr.hand_tracker, predicted_time).ok()??
            .map(|joint|
            {
                let pos = joint.pose.position;
                let rot = joint.pose.orientation;

                Transform
                {
                    pos: glam::vec3(pos.x, pos.y, pos.z),
                    rot: glam::quat(rot.x, rot.y, rot.z, rot.w),
                    size: glam::vec3(0.03, 0.03, 0.03)
                }
            });

        Some(Hand(res))
    }

    pub fn get_tip(&self, tip: usize) -> Option<Transform>
    {
        match tip
        {
            0 => Some(self[xr::HandJointEXT::THUMB_TIP]),
            1 => Some(self[xr::HandJointEXT::INDEX_TIP]),
            2 => Some(self[xr::HandJointEXT::MIDDLE_TIP]),
            3 => Some(self[xr::HandJointEXT::RING_TIP]),
            4 => Some(self[xr::HandJointEXT::LITTLE_TIP]),
            _ => None,
        }
    }

    pub fn tips(&self) -> [Transform; 5]
    {
        const TIPS : [xr::HandJointEXT; 5] = [
            xr::HandJointEXT::THUMB_TIP,
            xr::HandJointEXT::INDEX_TIP,
            xr::HandJointEXT::MIDDLE_TIP,
            xr::HandJointEXT::RING_TIP,
            xr::HandJointEXT::LITTLE_TIP,
        ];

        TIPS.map(|i| self.0[i])
    }

    pub fn compute_ray(&self) -> Ray
    {
        let middle_proximal = self[xr::HandJointEXT::MIDDLE_PROXIMAL].pos;
        let wrist = self[xr::HandJointEXT::WRIST].pos;
        let wrist_rot = self[xr::HandJointEXT::WRIST].rot;

        let start = (wrist + middle_proximal) / 2.0;
        let v = (wrist_rot * glam::Vec3::NEG_Z + wrist_rot * glam::Vec3::NEG_Y) / 2.0;

        return Ray::new_assume_normalize(start, v);
    }
}

impl Index<usize> for Hand
{
    type Output = Transform;

    fn index(&self, index: usize) -> &Self::Output
    {
        &self.0[index]
    }
}

impl Index<xr::HandJointEXT> for Hand
{
    type Output = Transform;

    fn index(&self, index: xr::HandJointEXT) -> &Self::Output
    {
        &self.0[index]
    }
}

impl GestureState
{
    pub fn new() -> Self
    {
        Self
        {
            previous_wrist: None,
            current_wrist: None
        }
    }

    pub fn update(&mut self, hand: &Option<Hand>, list: &ObjectList) -> Vec<Gesture>
    {
        self.previous_wrist = self.current_wrist;
        self.current_wrist = hand.as_ref().and_then(|hand| Some(hand[xr::HandJointEXT::WRIST]));

        if let Some(hand) = hand
        {
            return self.compute_gestures(hand, list);
        }
        else
        {
            return Vec::new();
        }
    }

    pub fn transform_since_last_frame(&self, obj_origin: &Vec3) -> Option<Transform>
    {
        let previous = self.previous_wrist?;
        let current = self.current_wrist?;

        let diff_pos = current.pos - previous.pos;
        let diff_rot = current.rot * previous.rot.inverse();
        let diff_scale = current.size / previous.size;

        let wrist_to_obj = obj_origin - current.pos;
        let wrist_rot_translation = diff_rot * wrist_to_obj - wrist_to_obj;

        Some(Transform
        {
            pos: diff_pos + wrist_rot_translation,
            rot: diff_rot,
            size: diff_scale
        })
    }

    fn compute_gestures(&mut self, hand: &Hand, list: &ObjectList) -> Vec<Gesture>
    {
        let mut gestures : Vec<Gesture> = Vec::new();

        let tip_obbs = hand.tips().map(|transform| OBB::CUBE_OBB.compute_obb(&transform));
        let ray = hand.compute_ray();

        let thumb_tip = hand[xr::HandJointEXT::THUMB_TIP].pos;
        let index_tip = hand[xr::HandJointEXT::INDEX_TIP].pos;

        let should_check_ray_hit = thumb_tip.distance_squared(index_tip) < 0.0003;
        let mut ray_hit : Option<(ObjectID, f32)> = None;

        'obj_it:for (id, obj) in list.iter()
        {
            let Some(obj_obb) = obj.compute_obb() else { continue; };

            // RAY

            if should_check_ray_hit
            {
                if let Some(dist) = obj_obb.does_intersects_ray(&ray)
                {
                    if let Some((_, current_dist)) = ray_hit
                    {
                        if dist < current_dist
                        {
                            ray_hit = Some((id, dist))
                        }
                    }
                    else
                    {
                        ray_hit = Some((id, dist))
                    }
                }
            }

            // GRAB

            let collisions = tip_obbs.map(|tip| tip.does_intersects_obb(&obj_obb));

            if collisions[HandTip::THUMB.to_idx()]
            {
                let mut is_grabbing = false;

                for tip in [HandTip::INDEX, HandTip::MIDDLE, HandTip::RING, HandTip::LITTLE]
                {
                    if collisions[tip.to_idx()]
                    {
                        is_grabbing = true;
                        break;
                    }
                }

                if is_grabbing
                {
                    gestures.push(Gesture::Drag { id });
                    continue 'obj_it;
                }
            }
        }

        if let Some((hit, _)) = ray_hit
        {
            let does_hit_not_have_gesture = gestures.iter().find(|g|
            {
                let &id = match g
                {
                    Gesture::Tap { id } => id,
                    Gesture::Drag { id } => id
                };

                hit == id
            }).is_none();

            if does_hit_not_have_gesture
            {
                gestures.push(Gesture::Drag { id: hit });
            }
        }

        return gestures;
    }
}
