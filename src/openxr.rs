use openxr::{self as xr, Extent2Di};
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

        Some(XRState
        {
            instance,
            system_id,
            environment_blend_mode,
            session,
            frame_waiter,
            frame_stream,
            stage,
            views
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
