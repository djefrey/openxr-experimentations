use std::{marker::PhantomData, ptr::null};

use openxr::{self as xr, CompositionLayerBase, Extent2Di, Hand, Passthrough, PassthroughFlagsFB, PassthroughLayerPurposeFB};
use vulkano::{swapchain, Handle, VulkanObject};

use crate::vulkan::VulkanState;

const VIEW_TYPE : xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

pub struct XRSetupState
{
    pub instance: xr::Instance,
    pub system_id: xr::SystemId,
    pub requirements: xr::vulkan::Requirements,
}

pub struct XRState
{
    pub instance: xr::Instance,
    pub system_id: xr::SystemId,
    pub environment_blend_mode: xr::EnvironmentBlendMode,
    pub session: xr::Session<xr::Vulkan>,

    pub frame_waiter: xr::FrameWaiter,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,

    // pub action_set: xr::ActionSet,
    // pub left_hand: xr::Action<xr::Posef>,
    // pub right_hand: xr::Action<xr::Posef>,

    // #[cfg(target_os = "android")]
    pub hand_tracker: xr::HandTracker,

    // pub left_space: xr::Space,
    // pub right_space: xr::Space,
    pub stage: xr::Space,

    pub views: Vec<xr::ViewConfigurationView>,
    // swapchain: xr::Swapchain<xr::Vulkan>,

    #[cfg(target_os = "android")]
    pub passthrough: XRPasstrough,

    pub depth_provider: xr::EnvironmentDepthProvider,
}

pub struct XRPasstrough
{
    pub passthrough: xr::Passthrough,
    pub layer: xr::PassthroughLayer
}

impl XRSetupState
{
    pub fn init() -> Option<Self>
    {
        let entry = unsafe { xr::Entry::load().ok()? };

        #[cfg(target_os = "android")]
        entry.initialize_android_loader().unwrap();

        let mut enabled_extensions = xr::ExtensionSet::default();
        enabled_extensions.khr_vulkan_enable2 = true;
        enabled_extensions.ext_hand_tracking = true;

        #[cfg(target_os = "android")]
        {
            enabled_extensions.khr_android_create_instance = true;
            enabled_extensions.fb_passthrough = true;
            enabled_extensions.fb_composition_layer_alpha_blend = true;
            enabled_extensions.meta_environment_depth = true;
        }

        let available_layers = entry.enumerate_layers().ok()?;

        println!("Availalble OpenXR Layers:");
        for layer in available_layers
        {
            println!("- {} ({})", layer.layer_name, layer.description);
        }

        let instance = entry.create_instance(&xr::ApplicationInfo
            {
                application_name: "openxr_tests",
                ..Default::default()
            }, &enabled_extensions,
            &[]).ok()?;

        let instance_props = instance.properties().ok()?;
        let system_id = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY).ok()?;
        let system_props = instance.system_properties(system_id).ok()?;

        println!("\nOpenXR Runtime: {} {}", instance_props.runtime_name, instance_props.runtime_version);

        let requirements = instance.graphics_requirements::<xr::Vulkan>(system_id).ok()?;

        println!("Vulkan Requirements: {} <= API VERSION <= {}", requirements.min_api_version_supported, requirements.max_api_version_supported);

        Some(XRSetupState
        {
            instance,
            system_id,
            requirements
        })
    }
}

impl XRState
{
    pub fn init(setup: XRSetupState, vulkan: &VulkanState) -> Option<Self>
    {
        let instance = setup.instance;
        let system_id = setup.system_id;

        println!("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");

        let environment_blend_mode = instance.enumerate_environment_blend_modes(system_id, VIEW_TYPE).ok()?[0];

        let (session, frame_waiter, frame_stream) = unsafe { instance.create_session::<xr::Vulkan>(system_id, &vulkan.to_session_create_infos()).ok()? };
        // let action_set = instance.create_action_set("input", "Input Action Set", 0).unwrap();

        // let left_hand = action_set.create_action::<xr::Posef>("left", "Left Hand", &[]).unwrap();
        // let right_hand = action_set.create_action::<xr::Posef>("right", "Right Hand", &[]).unwrap();

        // instance
        //     .suggest_interaction_profile_bindings(
        //         instance
        //             .string_to_path("/interaction_profiles/oculus/hand_tracking")
        //             .unwrap(),
        //         &[
        //             xr::Binding::new(
        //                 &left_hand,
        //                 instance
        //                     .string_to_path("/user/hand/left/input/grip/pose")
        //                     .unwrap(),
        //             ),
        //             xr::Binding::new(
        //                 &right_hand,
        //                 instance
        //                     .string_to_path("/user/hand/right/input/grip/pose")
        //                     .unwrap(),
        //             ),
        //         ],
        //     )
        //     .unwrap();

        // session.attach_action_sets(&[&action_set]).unwrap();

        // let left_space = left_hand
        //     .create_space(session.clone(), xr::Path::NULL, xr::Posef::IDENTITY)
        //     .unwrap();

        // let right_space = right_hand
        //     .create_space(session.clone(), xr::Path::NULL, xr::Posef::IDENTITY)
        //     .unwrap();

        // #[cfg(target_os = "android")]
        let right_hand_tracker = session.create_hand_tracker(Hand::RIGHT).unwrap();

        let stage = session.create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY).ok()?;
        let views = instance.enumerate_view_configuration_views(system_id, VIEW_TYPE).ok()?;

        #[cfg(target_os = "android")]
        let passthrough ={
            let passthrough = session.create_passthrough(xr::PassthroughFlagsFB::IS_RUNNING_AT_CREATION).unwrap();
            let layer = session.create_passthrough_layer(&passthrough, PassthroughFlagsFB::IS_RUNNING_AT_CREATION, PassthroughLayerPurposeFB::RECONSTRUCTION).unwrap();

            XRPasstrough
            {
                passthrough,
                layer
            }
        };

        let depth_provider = session.create_depth_environment_provider(xr::EnvironmentDepthProviderCreateFlagsMETA::EMPTY).unwrap();
        depth_provider.set_hand_removal(true);
        depth_provider.start().unwrap();

        Some(XRState
        {
            instance,
            system_id,
            environment_blend_mode,
            session,
            frame_waiter,
            frame_stream,
            // action_set,
            // left_hand,
            // right_hand,
            // left_space,
            // right_space,
            // #[cfg(target_os = "android")]
            hand_tracker: right_hand_tracker,
            stage,
            views,

            #[cfg(target_os = "android")]
            passthrough,

            depth_provider,
        })
    }
}

impl VulkanState
{
    pub fn to_session_create_infos(&self) -> xr::vulkan::SessionCreateInfo
    {
        xr::vulkan::SessionCreateInfo
        {
            instance: self.instance.handle().as_raw() as *const _,
            physical_device: self.pdevice.handle().as_raw() as *const _,
            device: self.device.handle().as_raw() as *const _,
            queue_family_index: self.queue_family_index,
            queue_index: 0,
        }
    }
}

pub struct CompositionLayerPassthroughFB<'a, G: xr::Graphics>
{
    inner: xr::sys::CompositionLayerPassthroughFB,
    // alpha_blend: Box<xr::sys::CompositionLayerAlphaBlendFB>,
    _marker: PhantomData<&'a G>,
}

impl<'a, G: xr::Graphics> CompositionLayerPassthroughFB<'a, G>
{
    pub fn new(layer: &xr::PassthroughLayer) -> Self
    {
        // let alpha_blend = Box::new(sys::CompositionLayerAlphaBlendFB
        // {
        //     ty: sys::StructureType::COMPOSITION_LAYER_ALPHA_BLEND_FB,
        //     next: null_mut(),
        //     src_factor_color: BlendFactorFB::ONE,
        //     dst_factor_color: BlendFactorFB::ZERO,
        //     src_factor_alpha: BlendFactorFB::ONE,
        //     dst_factor_alpha: BlendFactorFB::ONE
        // });

        // let ptr : *const sys::CompositionLayerAlphaBlendFB = &*alpha_blend;

        let inner = xr::sys::CompositionLayerPassthroughFB
        {
            ty: xr::sys::StructureType::COMPOSITION_LAYER_PASSTHROUGH_FB,
            next: null(), // ptr as *const _,
            flags: xr::sys::CompositionLayerFlags::EMPTY,
            space: xr::sys::Space::NULL,
            layer_handle: *layer.inner(),
        };

        Self { inner, _marker: PhantomData }
    }
}

impl<'a, G: xr::Graphics> std::ops::Deref for CompositionLayerPassthroughFB<'a, G>
{
    type Target = CompositionLayerBase<'a, G>;

    #[inline]
    fn deref(&self) -> &Self::Target
    {
        unsafe { std::mem::transmute(&self.inner) }
    }
}
