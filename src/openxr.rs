use std::{marker::PhantomData, ptr::null};

use openxr::{self as xr, CompositionLayerBase, Extent2Di, Passthrough, PassthroughFlagsFB, PassthroughLayerPurposeFB};
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

    // left_space: xr::Space,
    // right_space: xr::Space,
    pub stage: xr::Space,

    pub views: Vec<xr::ViewConfigurationView>,
    // swapchain: xr::Swapchain<xr::Vulkan>,

    #[cfg(target_os = "android")]
    pub passthrough: XRPasstrough,
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

        #[cfg(target_os = "android")]
        {
            enabled_extensions.khr_android_create_instance = true;
            enabled_extensions.fb_passthrough = true;
            enabled_extensions.fb_composition_layer_alpha_blend = true;
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

        let environment_blend_modes = instance.enumerate_environment_blend_modes(system_id, VIEW_TYPE).ok()?;

        println!("Blend Modes:");
        for blend_mode in &environment_blend_modes
        {
            println!("- {:?}", blend_mode);
        }

        let environment_blend_mode = environment_blend_modes[0];

        let (session, frame_waiter, frame_stream) = unsafe { instance.create_session::<xr::Vulkan>(system_id, &vulkan.to_session_create_infos()).ok()? };

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

        Some(XRState
        {
            instance,
            system_id,
            environment_blend_mode,
            session,
            frame_waiter,
            frame_stream,
            stage,
            views,

            #[cfg(target_os = "android")]
            passthrough
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
