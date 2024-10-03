uniform sampler2D u_texture;

in vec4 vs_color;
in vec2 vs_tex_coord;

out vec4 FragColor;

void main()
{
    FragColor = vs_color * texture(u_texture, vs_tex_coord).r;
}
