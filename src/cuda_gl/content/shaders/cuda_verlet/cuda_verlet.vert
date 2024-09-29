uniform mat3 u_world_to_view;

in vec2 a_vertex;
in vec2 a_tex_coord;
in vec4 a_color;
in vec2 a_position;
in vec2 a_scale;

out vec4 vs_color;
out vec2 vs_tex_coord;

void main()
{
    vec2 screen_pos = (u_world_to_view * vec3(a_position, 1)).xy;
    vec2 screen_size = (u_world_to_view * vec3(a_scale, 0)).xy;
    gl_Position = vec4(a_vertex * screen_size + screen_pos, 0.0, 1.0);
    vs_color = a_color;
    vs_tex_coord = a_tex_coord;
}
