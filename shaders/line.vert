
#version 450
#extension GL_EXT_multiview : require

layout(set = 0, binding = 0) uniform GlobalData
{
    mat4 view[2];
    mat4 proj[2];
    mat4 projView[2];
    mat4 invView[2];
    mat4 invProj[2];
} global;


layout(push_constant) uniform ObjectData
{
    mat4 transform;
    vec4 tint;
} object;

layout(location = 0) in vec3 position;
layout(location = 0) out vec4 outColor;

void main()
{
    gl_Position = global.projView[gl_ViewIndex] * object.transform * vec4(position, 1);
    outColor = object.tint;
}