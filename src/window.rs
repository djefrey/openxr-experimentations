use glam::{vec3, vec4, Quat, Vec2, Vec3};
use openxr::Vulkan;

use crate::{gestures::{Gesture, GestureKind, GesturePhase, GestureState, Hand, HandTip}, obb::OBB, object::{ObjectID, ObjectKind, ObjectList}, vulkan::{texture::VulkanTexture, RenderState, VulkanState}, Transform};

#[derive(Debug, Clone)]
pub struct Window
{
    obj_id: ObjectID,
    title: String,
    texture: VulkanTexture,
    is_hovering_close: bool
}

pub enum WindowEvent
{
    ContentInteract { tips: Vec<HandTip> },
    Drag,
    Close
}

impl Window
{
    const DECORATION_HEIGHT : f32 = 0.1;
    const CONTENT_HEIGHT : f32 = 0.4;
    const WINDOW_HEIGHT : f32 = Window::DECORATION_HEIGHT + Window::CONTENT_HEIGHT;
    const CLOSE_SIZE : f32 = 0.05;

    // From window center to element center
    const DECORATION_OFFSET : f32 = Window::WINDOW_HEIGHT / 2.0 - Window::DECORATION_HEIGHT / 2.0;
    const CONTENT_OFFSET : f32 = Window::DECORATION_OFFSET - Window::WINDOW_HEIGHT / 2.0;

    // Offset from Top Right
    const CLOSE_OFFSET_FROM_TR : Vec2 = Vec2 { x: Window::DECORATION_HEIGHT / 2.0, y: Window::DECORATION_HEIGHT / 2.0 };

    pub fn new(title: String, texture: VulkanTexture, pos: Vec3, rot: Quat, obj_list: &mut ObjectList) -> ObjectID
    {
        let transform = Transform::new(pos, rot, Vec3::ONE);
        let ratio = (texture.width as f32) / (texture.height as f32);

        let width = Window::WINDOW_HEIGHT * ratio;
        let height = Window::WINDOW_HEIGHT;

        let obj_id = obj_list.new_object(ObjectKind::Empty, transform, Some(OBB::new(vec3(width, height, 0.01))));

        let window = Window
        {
            obj_id,
            title,
            texture,
            is_hovering_close: false
        };

        let Some(obj) = obj_list.get_mut_object(obj_id) else { panic!("Window object is nil"); };
        obj.kind = ObjectKind::Window { window };

        return obj_id;
    }

    pub fn handle_gesture(&mut self, transform: &mut Transform, hand: &Hand, gesture: &Gesture, gestures: &GestureState) -> Option<WindowEvent>
    {
        let ratio = (self.texture.width as f32) / (self.texture.height as f32);
        let width = ratio * Window::WINDOW_HEIGHT;

        if let GesturePhase::Cancelled = gesture.phase
        {
            return None;
        }
        else if let GesturePhase::Ended { on_origin } = gesture.phase
        {
            if on_origin && self.is_hovering_close
            {
                return Some(WindowEvent::Close);
            }
        }
        else
        {
            // Interact with UI

            let close_offset = vec3(width / 2.0 - Window::CLOSE_OFFSET_FROM_TR.x, Window::WINDOW_HEIGHT / 2.0 - Window::CLOSE_OFFSET_FROM_TR.y, 0.0);
            let close_obb = OBB::new(vec3(Window::CLOSE_SIZE, Window::CLOSE_SIZE, 0.02))
                .compute_obb_with_offset(transform, close_offset);

            if let Some(gesture) = GestureState::compute_interaction_with_obb(close_obb, hand)
            {
                self.is_hovering_close = true;
            }
            else
            {
                self.is_hovering_close = false;
            }

            // Interact with Content

            let content_obb = OBB::new(vec3(width, Window::CONTENT_HEIGHT, 0.01))
                .compute_obb_with_offset(transform, vec3(0.0, Window::CONTENT_OFFSET, 0.0));

            if let Some(gesture) = GestureState::compute_interaction_with_obb(content_obb, hand)
            {
                // TODO: COMPUTE HIT LOCATION

                return match gesture
                {
                    GestureKind::Tap { tips } => Some(WindowEvent::ContentInteract { tips }),
                    GestureKind::Grab { tips } => Some(WindowEvent::ContentInteract { tips }),
                    GestureKind::Ray => Some(WindowEvent::ContentInteract { tips: vec![HandTip::THUMB, HandTip::MIDDLE, HandTip::LITTLE] }),
                }
            }

            // Move Window

            match gesture.kind
            {
                GestureKind::Grab { tips: _} | GestureKind::Ray =>
                {
                    if let Some(diff) = gestures.transform_since_last_frame(&transform.pos)
                    {
                        transform.pos += diff.pos;
                        transform.rot  = diff.rot * transform.rot;
                        // obj.transform.size *= diff.size;
                    }

                    return Some(WindowEvent::Drag);
                },
                GestureKind::Tap { tips: _ } => {},
            }
        }

        return None;
    }

    pub fn refresh_content(&mut self, texture: VulkanTexture, obb: &mut Option<OBB>)
    {
        let ratio = (texture.width as f32) / (texture.height as f32);

        let width = Window::WINDOW_HEIGHT * ratio;
        let height = Window::WINDOW_HEIGHT;

        self.texture = texture;
        *obb = Some(OBB::new(vec3(width, height, 0.01)));
    }
}

impl<'a> Window
{
    pub unsafe fn draw(&self, transform: &Transform, vk_state: &VulkanState, render_state: &'a mut RenderState)
    {
        let ratio = (self.texture.width as f32) / (self.texture.height as f32);
        let width = Window::WINDOW_HEIGHT * ratio;

        let deco_transform = Transform::new(
            transform.pos + transform.rot * vec3(0.0, Window::DECORATION_OFFSET, 0.0),
            transform.rot,
            transform.size * vec3(width, Window::DECORATION_HEIGHT, 0.01)
        );

        let close_offset = vec3(width / 2.0 - Window::CLOSE_OFFSET_FROM_TR.x, Window::WINDOW_HEIGHT / 2.0 - Window::CLOSE_OFFSET_FROM_TR.y, 0.0);

        let close_transform = Transform::new(
            transform.pos + transform.rot * close_offset,
            transform.rot,
            transform.size * vec3(Window::CLOSE_SIZE, Window::CLOSE_SIZE, 0.02)
        );

        let content_transform = Transform::new(
            transform.pos + transform.rot * vec3(0.0, Window::CONTENT_OFFSET, 0.0),
            transform.rot,
            transform.size * vec3(width, Window::CONTENT_HEIGHT, 0.01)
        );

        vk_state.render_tinted_cube(&deco_transform, &vec4(0.3, 0.3, 0.3, 0.3), render_state);
        vk_state.render_tinted_cube(&close_transform, &vec4(1.0, 0.2, 0.2, 0.8), render_state);
        vk_state.render_textured_quad(&content_transform, &self.texture, render_state);

        // vk_state.render_tinted_cube_wireframe(&deco_transform, &vec4(1.0, 0.0, 0.0, 0.66), render_state);
        // vk_state.render_tinted_cube_wireframe(&content_transform, &vec4(1.0, 0.0, 0.0, 0.66), render_state);

        // let close_offset = vec3(width / 2.0 - Window::CLOSE_OFFSET_FROM_TR.x, Window::WINDOW_HEIGHT / 2.0 - Window::CLOSE_OFFSET_FROM_TR.y, 0.0);
        // let close_obb = OBB::new(vec3(Window::CLOSE_SIZE, Window::CLOSE_SIZE, 0.02))
        //     .compute_obb_with_offset(transform, close_offset)
        //     .to_transform();

        // vk_state.render_tinted_cube_wireframe(&close_obb, &vec4(1.0, 0.0, 0.0, 0.66), render_state);

        // let content_obb = OBB::new(vec3(width, Window::CONTENT_HEIGHT, 0.01))
        //     .compute_obb_with_offset(transform, vec3(0.0, Window::CONTENT_OFFSET, 0.0))
        //     .to_transform();

        // vk_state.render_tinted_cube_wireframe(&content_obb, &vec4(1.0, 0.0, 0.0, 0.66), render_state);
    }
}
