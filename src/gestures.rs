use std::ops::Index;

use glam::Vec3;

use openxr::{self as xr};

use crate::{obb::{ComputedOOB, OBB}, object::{ObjectID, ObjectList}, openxr::XRState, ray::Ray, Transform};

#[derive(Debug, Clone)]
pub enum GestureKind
{
    Tap { tips: Vec<HandTip> },
    Grab { tips: Vec<HandTip> },
    Ray { dist: f32 },
}

impl PartialEq for GestureKind
{
    fn eq(&self, other: &Self) -> bool
    {
        match (self, other)
        {
            (Self::Tap { tips: _ } , Self::Tap { tips: _ }) => true,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GesturePhase
{
    Begin,
    Entered,
    Moved,
    Exited,
    Ended { on_origin: bool },
    Cancelled
}

#[derive(Debug, Clone)]
pub struct Gesture
{
    pub id: ObjectID,
    pub kind: GestureKind,
    pub phase: GesturePhase,
}

struct Interaction
{
    id: ObjectID,
    kind: GestureKind,
    has_leaved_origin: bool
}

pub struct GestureState
{
    previous_wrist: Option<Transform>,
    current_wrist: Option<Transform>,
    current_interaction: Option<Interaction>
}

#[derive(Debug, Clone, Copy)]
pub struct Hand([Transform; 26]);

#[derive(Debug, Clone, Copy)]
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
        let data = xr.stage.locate_hand_joints(&xr.hand_tracker, predicted_time).ok()??;

        fn is_joint_invalid<'a>(joint: &'a &xr::HandJointLocationEXT) -> bool
        {
            !(
                joint.location_flags.contains(xr::SpaceLocationFlags::POSITION_VALID)
                && joint.location_flags.contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
            )
        }

        if data.iter().find(is_joint_invalid).is_some()
        {
            return None;
        }

        let res = data.map(|joint|
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

        let start = wrist * 0.25 + middle_proximal * 0.75;
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
            current_wrist: None,
            current_interaction: None
        }
    }

    pub fn update(&mut self, hand: &Option<Hand>, list: &ObjectList) -> Vec<Gesture>
    {
        let mut res = Vec::new();

        self.previous_wrist = self.current_wrist;
        self.current_wrist = hand.as_ref().and_then(|hand| Some(hand[xr::HandJointEXT::WRIST]));

        if let Some(hand) = hand
        {
            if let Some(interaction) = &mut self.current_interaction
            {
                if let Some(new_kind) = GestureState::compute_interaction_with(interaction.id, hand, list)
                {
                    if interaction.kind == new_kind
                    {
                        // Continue current interaction

                        return vec![Gesture
                        {
                            id: interaction.id,
                            kind: new_kind,
                            phase: GesturePhase::Moved
                        }];
                    }
                    else
                    {
                        // Interaction kind updates

                        return vec![Gesture
                        {
                            id: interaction.id,
                            kind: interaction.kind.clone(),
                            phase: GesturePhase::Ended { on_origin: !interaction.has_leaved_origin }
                        },
                        Gesture
                        {
                            id: interaction.id,
                            kind: new_kind,
                            phase: GesturePhase::Begin
                        }];
                    }
                }
            }

            if let Some(new) = GestureState::check_for_new_interaction(hand, list)
            {
                let (id, ref kind) = new;
                let mut has_leaved_origin = false;

                if let Some(current_interaction) = self.current_interaction.take()
                {
                    has_leaved_origin = current_interaction.has_leaved_origin;

                    if current_interaction.kind == *kind
                    {
                        // Same interaction, different object

                        res.push(Gesture
                        {
                            id: current_interaction.id,
                            kind: kind.clone(),
                            phase: GesturePhase::Exited
                        });

                        res.push(Gesture
                        {
                            id,
                            kind: kind.clone(),
                            phase: GesturePhase::Entered
                        });

                        has_leaved_origin = true;
                    }
                    else
                    {
                        // Different interaction

                        res.push(Gesture
                        {
                            id: current_interaction.id,
                            kind: kind.clone(),
                            phase: GesturePhase::Ended { on_origin: !has_leaved_origin }
                        });

                        res.push(Gesture
                        {
                            id,
                            kind: kind.clone(),
                            phase: GesturePhase::Begin
                        });
                    }
                }
                else
                {
                    // New interaction

                    res.push(Gesture
                    {
                        id,
                        kind: kind.clone(),
                        phase: GesturePhase::Begin
                    });
                }

                self.current_interaction = Some(Interaction { id: new.0, kind: new.1, has_leaved_origin });
            }
            else if let Some(end_gesture) = self.end_current_interaction(false)
            {
                // No current interaction

                res.push(end_gesture);
            }
        }
        else
        {
            // Hand lost

            if let Some(end_gesture) = self.end_current_interaction(true)
            {
                res.push(end_gesture);
            }
        }

        return res;
    }

    fn end_current_interaction(&mut self, cancelled: bool) -> Option<Gesture>
    {
        let current = self.current_interaction.take()?;

        return Some(Gesture
        {
            id: current.id,
            kind: current.kind,
            phase: if !cancelled { GesturePhase::Ended { on_origin: !current.has_leaved_origin } } else { GesturePhase::Cancelled }
        });
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

    pub fn check_for_new_interaction(hand: &Hand, list: &ObjectList) -> Option<(ObjectID, GestureKind)>
    {
        let tip_obbs = hand.tips().map(|transform| OBB::CUBE_OBB.compute_obb(&transform));
        let ray = hand.compute_ray();

        let thumb_tip = hand[xr::HandJointEXT::THUMB_TIP].pos;
        let index_tip = hand[xr::HandJointEXT::INDEX_TIP].pos;

        let should_check_ray_hit = thumb_tip.distance_squared(index_tip) < 0.0003;
        let mut ray_hit : Option<(ObjectID, f32)> = None;

        for (id, obj) in list.iter()
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

            let tips = collisions.into_iter()
                .enumerate()
                .filter(|(_, does_collide)| *does_collide)
                .map(|(i, _)| HandTip(i))
                .collect::<Vec<HandTip>>();

            if collisions[HandTip::THUMB.to_idx()]
            {
                for tip in [HandTip::INDEX, HandTip::MIDDLE, HandTip::RING, HandTip::LITTLE]
                {
                    if collisions[tip.to_idx()]
                    {
                        return Some((id, GestureKind::Grab { tips }));
                    }
                }
            }

            if collisions.contains(&true)
            {
                return Some((id, GestureKind::Tap { tips }))
            }
        }

        if let Some((id, dist)) = ray_hit
        {
            return Some((id, GestureKind::Ray { dist }));
        }

        return None;
    }

    pub fn compute_interaction_with(id: ObjectID, hand: &Hand, list: &ObjectList) -> Option<GestureKind>
    {
        let obj = list.get_object(id)?;
        let obj_obb = obj.compute_obb()?;

        return GestureState::compute_interaction_with_obb(obj_obb, hand);
    }

    pub fn compute_interaction_with_obb(obb: ComputedOOB, hand: &Hand) -> Option<GestureKind>
    {
        let tip_obbs = hand.tips().map(|transform| OBB::CUBE_OBB.compute_obb(&transform));
        let collisions = tip_obbs.map(|tip| tip.does_intersects_obb(&obb));

        let tips = collisions.into_iter()
            .enumerate()
            .filter(|(_, does_collide)| *does_collide)
            .map(|(i, _)| HandTip(i))
            .collect::<Vec<HandTip>>();

        if collisions[HandTip::THUMB.to_idx()]
        {
            for tip in [HandTip::INDEX, HandTip::MIDDLE, HandTip::RING, HandTip::LITTLE]
            {
                if collisions[tip.to_idx()]
                {
                    return Some(GestureKind::Grab { tips });
                }
            }
        }

        if collisions.contains(&true)
        {
            return Some(GestureKind::Tap { tips })
        }

        let ray = hand.compute_ray();

        let thumb_tip = hand[xr::HandJointEXT::THUMB_TIP].pos;
        let index_tip = hand[xr::HandJointEXT::INDEX_TIP].pos;

        if thumb_tip.distance_squared(index_tip) < 0.0003
        {
            if let Some(dist) = obb.does_intersects_ray(&ray)
            {
                return Some(GestureKind::Ray { dist });
            }
        }

        return None;
    }
}
