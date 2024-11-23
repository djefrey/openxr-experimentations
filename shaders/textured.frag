#version 450

layout(set = 1, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec4 inTint;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(tex, inUV) * inTint;
}
