// One quad per Verlet object. The CUDA-written object buffer is bound as an
// instance-rate vertex buffer, while the six quad corners come from the vertex
// index.

struct VertexInput
{
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float4 color;
    [[vk::location(2)]] float2 scale;
};

struct PushConstants
{
    float4 col0;  // columns of the world-to-view matrix
    float4 col1;
    float4 col2;
};

[[vk::push_constant]] PushConstants pc;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
    [[vk::location(1)]] float2 tex_coord : TEXCOORD0;
};

static const float2 kCorners[6] = {
    float2(-1, -1), float2(1, -1), float2(1, 1),
    float2(-1, -1), float2(1, 1), float2(-1, 1),
};

[shader("vertex")]
VertexOutput main(VertexInput input, uint vertex_index: SV_VertexID)
{
    const float2 corner = kCorners[vertex_index];
    const float2 screen_position =
        pc.col0.xy * input.position.x + pc.col1.xy * input.position.y + pc.col2.xy;
    const float2 screen_size =
        pc.col0.xy * input.scale.x + pc.col1.xy * input.scale.y;

    VertexOutput output;
    output.position = float4(corner * screen_size + screen_position, 0.0, 1.0);
    output.color = input.color;
    output.tex_coord = corner * 0.5 + 0.5;
    return output;
}
