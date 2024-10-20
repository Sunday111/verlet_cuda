#pragma once

#include <fmt/chrono.h>

#include <EverydayTools/Math/FloatRange.hpp>
#include <EverydayTools/Math/Math.hpp>
#include <klgl/mesh/mesh_data.hpp>
#include <klgl/mesh/procedural_mesh_generator.hpp>

#include "cuda_util.hpp"
#include "imgui.h"
#include "kernels.hpp"
#include "klgl/application.hpp"
#include "klgl/camera/camera_2d.hpp"
#include "klgl/events/event_listener_interface.hpp"
#include "klgl/events/mouse_events.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/opengl/vertex_attribute_helper.hpp"
#include "klgl/shader/shader.hpp"
#include "klgl/texture/texture.hpp"
#include "klgl/window.hpp"

namespace verlet
{

class SpawnColorStrategy;
class Emitter;

class VerletCudaApp : public klgl::Application
{
public:
    using GL = klgl::OpenGl;
    using ColorType = Vec4f;
    using ColorAttribHelper = klgl::VertexBufferHelperStatic<ColorType>;
    using PositionType = Vec2f;
    using PositionAttribHelper = klgl::VertexBufferHelperStatic<PositionType>;
    using ScaleType = Vec2f;
    using ScaleAttribHelper = klgl::VertexBufferHelperStatic<ScaleType>;

    static constexpr unsigned kSeed = 12345;

    VerletCudaApp();
    ~VerletCudaApp() override;

    void Initialize() override;
    void RegisterGLBuffers();
    void CreateMesh();
    void CreateCircleMaskTexture();
    void UpdateCamera();
    void OnMouseScroll(const klgl::events::OnMouseScroll& event);
    Vec2f GetMousePositionInWorldCoordinates() const;
    void Tick() override;
    void AddObject(const VerletObject& object)
    {
        if ((used_objects_count_ + pending_objects_.size()) < GetMaxObjectsCount())
        {
            pending_objects_.push_back(object);
        }
    }

    [[nodiscard]] size_t GetMaxObjectsCount() const { return 1'500'000; }
    [[nodiscard]] size_t GetObjectsCount() const { return used_objects_count_; }
    [[nodiscard]] SpawnColorStrategy& GetSpawnColorStrategy() const { return *spawn_color_strategy_; }

private:
    void SpawnPendingObjects();
    std::tuple<CudaMappedGraphicsResourcePtr, std::span<VerletObject>> ReserveAndGetDevicePtr(size_t required_size);

private:
    cudaStream_t cuda_stream_{};

    std::unique_ptr<klgl::events::IEventListener> event_listener_;
    std::unique_ptr<SpawnColorStrategy> spawn_color_strategy_;

    float zoom_power_ = 0.f;
    klgl::Camera2d camera_{};
    klgl::Viewport viewport_{};
    klgl::RenderTransforms2d render_transforms_{};
    std::shared_ptr<klgl::Shader> shader_;

    size_t a_vertex_{};
    size_t a_tex_coord_{};
    size_t a_color_{};
    size_t a_position_{};
    size_t a_scale_{};

    klgl::GlObject<klgl::GlBufferId> objects_vbo_;
    CudaGraphicsResourcePtr objects_vbo_resource_;

    std::shared_ptr<klgl::MeshOpenGL> mesh_;

    klgl::UniformHandle u_texture_{"u_texture"};
    std::unique_ptr<klgl::Texture> texture_;

    klgl::UniformHandle u_world_to_view_{"u_world_to_view"};

    CudaPtr<GridCell> grid_cells_;

    std::vector<VerletObject> pending_objects_;
    std::vector<std::unique_ptr<Emitter>> emitters_;

    size_t reserved_objects_count_ = 0;
    size_t used_objects_count_ = 0;

    ImFont* big_font_ = nullptr;
};
}  // namespace verlet
