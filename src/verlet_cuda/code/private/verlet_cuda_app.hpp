#pragma once

#include <fmt/chrono.h>

#include <EverydayTools/Math/FloatRange.hpp>
#include <EverydayTools/Math/Math.hpp>

#include "cuda_util.hpp"
#include "imgui.h"
#include "kernels.hpp"
#include "klvk/application.hpp"
#include "klvk/camera/camera_2d.hpp"
#include "klvk/events/event_listener_interface.hpp"
#include "klvk/events/mouse_events.hpp"
#include "klvk/vulkan/descriptor_sets.hpp"
#include "klvk/vulkan/texture.hpp"
#include "klvk/vulkan/vk_object.hpp"
#include "klvk/window.hpp"

namespace verlet
{

class SpawnColorStrategy;
class Emitter;

class VerletCudaApp : public klvk::Application
{
public:
    static constexpr unsigned kSeed = 12345;

    VerletCudaApp();
    ~VerletCudaApp() override;

    void Initialize() override;
    void CreateCircleMaskTexture();
    void CreatePipeline();
    void UpdateCamera();
    void OnMouseScroll(const klvk::events::OnMouseScroll& event);
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
    // Grows the shared buffer when needed and returns a device view of the objects.
    std::span<VerletObject> ReserveAndGetDevicePtr(size_t required_size);
    void DrawObjects();

private:
    cudaStream_t cuda_stream_{};

    std::unique_ptr<klvk::events::IEventListener> event_listener_;
    std::unique_ptr<SpawnColorStrategy> spawn_color_strategy_;

    float zoom_power_ = 0.f;
    klvk::Camera2d camera_{};
    klvk::Viewport viewport_{};
    klvk::RenderTransforms2d render_transforms_{};

    // The objects live in memory shared with CUDA: kernels write it, the vertex
    // shader reads it as an instance-rate vertex buffer.
    CudaVkBuffer objects_buffer_;

    std::unique_ptr<klvk::Texture> texture_;
    klvk::DescriptorSets descriptor_sets_;
    klvk::VkObject<VkPipelineLayout> pipeline_layout_;
    klvk::VkObject<VkPipeline> pipeline_;

    CudaPtr<GridCell> grid_cells_;

    std::vector<VerletObject> pending_objects_;
    std::vector<std::unique_ptr<Emitter>> emitters_;

    size_t reserved_objects_count_ = 0;
    size_t used_objects_count_ = 0;

    ImFont* big_font_ = nullptr;
};
}  // namespace verlet
