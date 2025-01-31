#version 450
#extension GL_EXT_multiview : require

layout(set = 0, binding = 0) uniform GlobalData
{
    mat4 proj[2];
    mat4 invProj[2];
} global;

layout(push_constant) uniform ObjectData
{
    mat4 transform;
    vec4 tint;
} object;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 outTint;

void main()
{
    gl_Position = global.proj[gl_ViewIndex] * object.transform * vec4(position, 1);
    outUV = uv;
    outTint = object.tint;
}
