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

layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput depthInput;

layout(push_constant) uniform CursorData
{
    float radius;
    float size;
    vec2 _padding;
    vec3 pos;
    float _padding2;
} cursor;

layout(location = 0) in vec2 xy;
layout(location = 0) out vec4 outColor;

void main()
{
    float depth = subpassLoad(depthInput).r;
    vec4 clipSpacePos = vec4(xy, depth, 1.0);

    vec4 viewSpacePos = global.invProj[gl_ViewIndex] * clipSpacePos;
    viewSpacePos /= viewSpacePos.w;

    vec4 worldSpacePos = global.invView[gl_ViewIndex] * viewSpacePos;

    float dist = distance(worldSpacePos.xyz, cursor.pos);
    float alpha = max(exp(1.0 - (abs(dist - cursor.radius) / cursor.size)), 0.0);

    outColor = vec4(0.1, 0.1, 0.1, alpha);
}
