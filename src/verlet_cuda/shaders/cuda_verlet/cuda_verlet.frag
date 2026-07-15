#version 450

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(location = 0) in vec4 vs_color;
layout(location = 1) in vec2 vs_tex_coord;

layout(location = 0) out vec4 FragColor;

void main()
{
    FragColor = vs_color * texture(u_texture, vs_tex_coord).r;
}
