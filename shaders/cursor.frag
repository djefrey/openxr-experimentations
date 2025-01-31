#version 450
#extension GL_EXT_multiview : require

layout(set = 0, binding = 0) uniform GlobalData
{
    mat4 proj[2];
    mat4 invProj[2];
} global;

layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput depthInput;

layout(location = 0) in vec2 xy;
layout(location = 0) out vec4 outColor;

void main()
{
    float depth = subpassLoad(depthInput).r;
    vec4 pos = global.invProj[gl_ViewIndex] * vec4(xy, depth, 1.0);

    outColor = vec4(1.0, 0.0, 0.0, 1.0 - depth);
}
