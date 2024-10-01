#pragma once

#include <cuda_runtime.h>
#include <fmt/chrono.h>

#include <EverydayTools/Math/FloatRange.hpp>
#include <EverydayTools/Math/Math.hpp>
#include <klgl/mesh/mesh_data.hpp>
#include <klgl/mesh/procedural_mesh_generator.hpp>

#include "camera.hpp"
#include "kernels.hpp"
#include "klgl/application.hpp"
#include "klgl/events/event_listener_interface.hpp"
#include "klgl/events/mouse_events.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/opengl/vertex_attribute_helper.hpp"
#include "klgl/reflection/matrix_reflect.hpp"  // IWYU pragma: keep
#include "klgl/shader/shader.hpp"
#include "klgl/texture/texture.hpp"
#include "klgl/window.hpp"
#include "object.hpp"

namespace verlet
{

using CudaDeleter = decltype([](auto* p) { cudaFree(p); });

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

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
    void SpawnPendingObjects();
    void CreateMesh();
    void CreateCircleMaskTexture();
    void UpdateCamera();
    void OnMouseScroll(const klgl::events::OnMouseScroll& event);
    void UpdateRenderTransforms();
    Vec2f GetMousePositionInWorldCoordinates() const;
    void Tick() override;
    void CheckGrid(std::span<PositionType> device_positions);
    void AddObject(const VerletObject& object);

    [[nodiscard]] size_t GetMaxObjectsCount() const { return 3'000'000; }
    [[nodiscard]] size_t GetObjectsCount() const { return used_objects_count_; }
    [[nodiscard]] SpawnColorStrategy& GetSpawnColorStrategy() const { return *spawn_color_strategy_; }

private:
    cudaStream_t cuda_stream_{};

    std::unique_ptr<klgl::events::IEventListener> event_listener_;
    std::unique_ptr<SpawnColorStrategy> spawn_color_strategy_;

    Camera camera_{};
    std::shared_ptr<klgl::Shader> shader_;

    size_t a_vertex_{};
    size_t a_tex_coord_{};

    size_t a_color_{};
    klgl::GlObject<klgl::GlBufferId> color_vbo_;

    size_t a_position_{};
    klgl::GlObject<klgl::GlBufferId> position_vbo_;

    CudaPtr<PositionType> device_old_positions_;

    size_t a_scale_{};
    klgl::GlObject<klgl::GlBufferId> scale_vbo_;
    std::vector<ScaleType> scales_;

    std::shared_ptr<klgl::MeshOpenGL> mesh_;

    klgl::UniformHandle u_texture_{"u_texture"};
    std::unique_ptr<klgl::Texture> texture_;

    klgl::UniformHandle u_world_to_view_{"u_world_to_view"};
    Mat3f world_to_view_ = Mat3f::Identity();

    Mat3f world_to_camera_ = Mat3f::Identity();
    Mat3f screen_to_world_ = Mat3f::Identity();

    CudaPtr<GridCell> grid_cells_;

    std::vector<Vec2f> pending_positions_;
    std::vector<Vec2f> pending_old_positions_;
    std::vector<Vec4f> pending_colors_;
    std::vector<Vec2f> pending_scales_;
    std::vector<std::unique_ptr<Emitter>> emitters_;

    size_t reserved_objects_count_ = 0;
    size_t used_objects_count_ = 0;
};
}  // namespace verlet
