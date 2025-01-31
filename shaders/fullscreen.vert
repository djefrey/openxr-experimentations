#version 450

layout(location = 0) out vec2 xy;

void main()
{
    float x = float((gl_VertexIndex & 1) * 2 - 1); // Maps (0,1,2,3) → (-1,1,-1,1)
    float y = float(((gl_VertexIndex >> 1) & 1) * 2 - 1); // Maps (0,1,2,3) → (-1,-1,1,1)

    gl_Position = vec4(x, y, 0.0, 1.0);
    xy = vec2(x, y);
}
