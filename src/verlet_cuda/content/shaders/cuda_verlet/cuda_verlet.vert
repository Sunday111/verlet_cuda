#version 450

// One quad per verlet object. The quad corners are generated from gl_VertexIndex;
// the per-instance attributes are read straight out of the buffer the CUDA kernels
// write, which is bound here as an instance-rate vertex buffer.

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_scale;

layout(push_constant) uniform PushConstants
{
    vec4 col0;  // columns of the world-to-view matrix
    vec4 col1;
    vec4 col2;
} pc;

layout(location = 0) out vec4 vs_color;
layout(location = 1) out vec2 vs_tex_coord;

const vec2 kCorners[6] = vec2[](
    vec2(-1, -1), vec2(1, -1), vec2(1, 1),
    vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main()
{
    mat3 world_to_view = mat3(pc.col0.xyz, pc.col1.xyz, pc.col2.xyz);
    vec2 corner = kCorners[gl_VertexIndex];

    vec2 screen_pos = (world_to_view * vec3(a_position, 1)).xy;
    vec2 screen_size = (world_to_view * vec3(a_scale, 0)).xy;
    gl_Position = vec4(corner * screen_size + screen_pos, 0.0, 1.0);

    vs_color = a_color;
    vs_tex_coord = corner * 0.5 + 0.5;
}
