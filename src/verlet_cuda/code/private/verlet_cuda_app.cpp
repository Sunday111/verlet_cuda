#include "verlet_cuda_app.hpp"

#include "coloring/spawn_color/spawn_color_strategy_rainbow.hpp"
#include "constants.hpp"
#include "emitters/radial_emitter.hpp"
#include "imgui.h"
#include "klvk/error_handling.hpp"
#include "klvk/events/event_listener_method.hpp"
#include "klvk/events/event_manager.hpp"
#include "klvk/reflection/matrix_reflect.hpp"  // IWYU pragma: keep
#include "klvk/texture/procedural_texture_generator.hpp"
#include "klvk/vulkan/device_context.hpp"
#include "klvk/vulkan/graphics_pipeline_builder.hpp"
#include "klvk/vulkan/vulkan_api.hpp"

// Vulkan create-info structs are designed for partial designated initialization;
// unlisted fields must be zero.
#ifdef __clang__
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace verlet
{
namespace
{

// The world-to-view matrix as three vec4 columns, matching the push constant block layout.
struct PushConstants
{
    std::array<Vec4f, 3> columns;
};

}  // namespace

[[nodiscard]] static constexpr ImVec2 ToImVec(Vec2f v) noexcept
{
    return {v.x(), v.y()};
}

[[nodiscard]] static constexpr Vec2f FromImVec(ImVec2 v) noexcept
{
    return {v.x, v.y};
}

[[nodiscard]] inline auto MeasureTime(auto&& f)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float, std::milli>;
    auto start_time = Clock::now();
    f();
    auto finish_time = Clock::now();
    return std::chrono::duration_cast<Duration>(finish_time - start_time);
}

VerletCudaApp::VerletCudaApp()
{
    event_listener_ = klvk::events::EventListenerMethodCallbacks<&VerletCudaApp::OnMouseScroll>::CreatePtr(this);
    GetEventManager().AddEventListener(*event_listener_);
    spawn_color_strategy_ = std::make_unique<SpawnColorStrategyRainbow>(*this);
}

VerletCudaApp::~VerletCudaApp() = default;

void VerletCudaApp::Initialize()
{
    klvk::Application::Initialize();

    SetTargetFramerate({60.f});

    SetClearColor({});
    GetWindow().SetSize(1920, 1080);
    GetWindow().SetTitle("Cuda and Vulkan");

    CreateCircleMaskTexture();
    CreatePipeline();

    CheckResult(cudaStreamCreate(&cuda_stream_));

    grid_cells_ = MakeCudaArray<GridCell>(constants::kGridNumCells, cuda_stream_);
    CudaMemset(std::span{grid_cells_.get(), constants::kGridNumCells}, 0, cuda_stream_);
    cudaStreamSynchronize(cuda_stream_);

    big_font_ = [&](float pixel_size)
    {
        ImGuiIO& io = ImGui::GetIO();
        ImFontConfig config;
        config.SizePixels = pixel_size;
        config.OversampleH = config.OversampleV = 1;
        config.PixelSnapH = true;
        ImFont* font = io.Fonts->AddFontDefault(&config);
        return font;
    }(45);

    do  // NOLINT
    {
        zoom_power_ -= 0.1f;
        camera_.zoom = std::max(std::powf(1.1f, zoom_power_), std::numeric_limits<float>::lowest());
    } while (camera_.zoom > 1.f / constants::kWorldRange.Extent().Max());
}

void VerletCudaApp::CreateCircleMaskTexture()
{
    constexpr auto size = Vec2<size_t>{} + 128;
    const auto pixels = klvk::ProceduralTextureGenerator::CircleMask(size, 2);
    texture_ = klvk::Texture::CreateR8(GetDeviceContext(), size.Cast<uint32_t>(), pixels);
}

void VerletCudaApp::CreatePipeline()
{
    klvk::DeviceContext& context = GetDeviceContext();
    const VkDevice device = context.GetDevice();

    descriptor_sets_ = klvk::DescriptorSets::Builder(context)
                           .Binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                           .Build(1);
    descriptor_sets_.WriteImage(0, 0, texture_->GetView(), texture_->GetSampler());

    {
        const std::array push_constant_ranges{VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        }};
        const std::array set_layouts{descriptor_sets_.GetLayout()};
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = set_layouts.size(),
            .pSetLayouts = set_layouts.data(),
            .pushConstantRangeCount = push_constant_ranges.size(),
            .pPushConstantRanges = push_constant_ranges.data(),
        };
        pipeline_layout_ =
            klvk::VkObject<VkPipelineLayout>{device, klvk::Vulkan::CreatePipelineLayout(device, layout_info)};
    }

    // The CUDA-written objects buffer is bound as an instance-rate vertex buffer, so the
    // shader reads exactly the memory the kernels wrote.
    pipeline_ = klvk::VkObject<VkPipeline>{
        device,
        klvk::GraphicsPipelineBuilder(*this)
            .Layout(pipeline_layout_)
            .VertexShaderFile(GetShaderDir() / "cuda_verlet/cuda_verlet.vert")
            .FragmentShaderFile(GetShaderDir() / "cuda_verlet/cuda_verlet.frag")
            .VertexBinding(0, sizeof(VerletObject), VK_VERTEX_INPUT_RATE_INSTANCE)
            .VertexAttribute(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(VerletObject, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VerletObject, color))
            .VertexAttribute(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(VerletObject, scale))
            .AlphaBlend()
            .Build()};
}

std::span<VerletObject> VerletCudaApp::ReserveAndGetDevicePtr(size_t required_size)
{
    if (required_size <= reserved_objects_count_)
    {
        return objects_buffer_.GetDeviceSpan<VerletObject>();
    }

    constexpr size_t kCapacityGrowthStep = 5000;
    const size_t new_capacity = kCapacityGrowthStep * ((required_size + kCapacityGrowthStep) / kCapacityGrowthStep);
    CudaVkBuffer new_buffer(GetDeviceContext(), sizeof(VerletObject) * new_capacity);
    const std::span<VerletObject> new_device_objects = new_buffer.GetDeviceSpan<VerletObject>();

    if (used_objects_count_ != 0)
    {
        const std::span<VerletObject> prev_device_objects = objects_buffer_.GetDeviceSpan<VerletObject>();
        klvk::ErrorHandling::Ensure(
            prev_device_objects.size() == reserved_objects_count_,
            "Unexpected size of device array. Expected: {}. Actual: {}",
            reserved_objects_count_,
            prev_device_objects.size());
        CudaMemcpy(
            prev_device_objects.subspan(0, used_objects_count_),
            new_device_objects.subspan(0, used_objects_count_),
            cudaMemcpyDeviceToDevice,
            cuda_stream_);
        CheckResult(cudaStreamSynchronize(cuda_stream_));
    }

    // Destroying the old buffer here is only safe because Tick made the device idle
    // before calling into this, so no in-flight frame is still reading it.
    objects_buffer_ = std::move(new_buffer);

    reserved_objects_count_ = new_capacity;
    return new_device_objects;
}

void VerletCudaApp::SpawnPendingObjects()
{
    if (pending_objects_.empty())
    {
        return;
    }

    const size_t new_count = used_objects_count_ + pending_objects_.size();
    const std::span<VerletObject> device_objects = ReserveAndGetDevicePtr(new_count);
    klvk::ErrorHandling::Ensure(
        device_objects.size() == reserved_objects_count_,
        "Unexpected size of device array. Expected: {}. Actual: {}",
        reserved_objects_count_,
        device_objects.size());
    CudaMemcpy(
        std::span{pending_objects_},
        device_objects.subspan(used_objects_count_, pending_objects_.size()),
        cudaMemcpyHostToDevice,
        cuda_stream_);

    CheckResult(cudaStreamSynchronize(cuda_stream_));

    used_objects_count_ = new_count;
    pending_objects_.clear();
}

void VerletCudaApp::UpdateCamera()
{
    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        Vec2f offset{};
        if (ImGui::IsKeyDown(ImGuiKey_W)) offset.y() += 1.f;
        if (ImGui::IsKeyDown(ImGuiKey_S)) offset.y() -= 1.f;
        if (ImGui::IsKeyDown(ImGuiKey_D)) offset.x() += 1.f;
        if (ImGui::IsKeyDown(ImGuiKey_A)) offset.x() -= 1.f;

        camera_.eye += GetLastFrameDurationSeconds() * offset / camera_.zoom;
    }

    viewport_.MatchWindowSize(GetWindow().GetSize());
    render_transforms_.Update(camera_, viewport_, klvk::AspectRatioPolicy::ShrinkToFit);
}

void VerletCudaApp::OnMouseScroll(const klvk::events::OnMouseScroll& event)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        zoom_power_ += event.value.y();
        camera_.zoom = std::max(std::powf(1.1f, zoom_power_), std::numeric_limits<float>::lowest());
    }
}

Vec2f VerletCudaApp::GetMousePositionInWorldCoordinates() const
{
    auto p = FromImVec(ImGui::GetMousePos());
    p.y() = GetWindow().GetSize2f().y() - p.y();
    return edt::Math::TransformPos(render_transforms_.screen_to_world, p);
}

void VerletCudaApp::DrawObjects()
{
    if (used_objects_count_ == 0) return;

    const VkCommandBuffer command_buffer = GetCurrentCommandBuffer();
    const std::array descriptor_sets{descriptor_sets_.Get(0)};
    klvk::Vulkan::CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    klvk::Vulkan::CmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout_,
        0,
        descriptor_sets);

    const std::array vertex_buffers{objects_buffer_.GetHandle()};
    const std::array<VkDeviceSize, 1> offsets{0};
    klvk::Vulkan::CmdBindVertexBuffers(command_buffer, 0, vertex_buffers, offsets);

    // The shader constructs the mat3 from columns.
    PushConstants push_constants{};
    for (size_t column = 0; column != 3; ++column)
    {
        const Vec3f matrix_column = render_transforms_.world_to_view.GetColumn(column);
        push_constants.columns[column] = Vec4f{matrix_column.x(), matrix_column.y(), matrix_column.z(), 0.f};
    }
    klvk::Vulkan::CmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, push_constants);

    klvk::Vulkan::CmdDraw(command_buffer, 6, static_cast<uint32_t>(used_objects_count_), 0, 0);
}

void VerletCudaApp::Tick()
{
    klvk::Application::Tick();

    UpdateCamera();

    // CUDA and Vulkan share the objects buffer but are not synchronized through external
    // semaphores here, so frames still in flight could be reading it while the kernels
    // write. Growing the buffer also destroys the old VkBuffer. Going idle first covers
    // both, at the cost of the frame overlap the OpenGL version got from the driver.
    GetDeviceContext().WaitIdle();

    // Objects queued during the previous tick become visible now. This ran at the end of
    // the tick under OpenGL; it has to happen before the draw is recorded because a
    // reallocation swaps the VkBuffer the draw would reference.
    SpawnPendingObjects();

    // Do simulation substeps
    auto sim_time = MeasureTime(
        [&]
        {
            if (used_objects_count_ == 0) return;

            std::span<VerletObject> device_objects = objects_buffer_.GetDeviceSpan<VerletObject>();
            klvk::ErrorHandling::Ensure(
                device_objects.size() == reserved_objects_count_,
                "Unexpected size of positions array on device. Expected: {}. Actual: {}",
                reserved_objects_count_,
                device_objects.size());

            // Now it is a view to an array of full capacity. We need only subset of it
            device_objects = device_objects.subspan(0, used_objects_count_);

            for (size_t substep = 0; substep != constants::kNumSubSteps; ++substep)
            {
                CheckResult(
                    cudaMemsetAsync(grid_cells_.get(), 255, sizeof(GridCell) * constants::kGridNumCells, cuda_stream_));
                Kernels::PopulateGrid(cuda_stream_, grid_cells_.get(), device_objects.data(), device_objects.size());

                for (size_t offset_y = 0; offset_y != 3; ++offset_y)
                {
                    for (size_t offset_x = 0; offset_x != 3; ++offset_x)
                    {
                        Kernels::SolveCollisions(
                            cuda_stream_,
                            grid_cells_.get(),
                            device_objects.data(),
                            {offset_x, offset_y});
                    }
                }
                Kernels::UpdatePositions(cuda_stream_, used_objects_count_, device_objects.data());
            }

            cudaStreamSynchronize(cuda_stream_);
        });

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    constexpr ImVec2 kDebugWindowSize{640.0f, 840.0f};
    ImGui::SetNextWindowSize(kDebugWindowSize, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2{viewport->WorkPos.x + 32.0f, viewport->WorkPos.y + 32.0f}, ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Debug"))
    {
        if (ImGui::CollapsingHeader("Perf"))
        {
            std::string txt = fmt::format("{}", sim_time);
            ImGui::Text("%s", txt.data());                                      // NOLINT
            ImGui::Text("Framerate: %f", static_cast<double>(GetFramerate()));  // NOLINT
        }

        if (ImGui::CollapsingHeader("Emitters"))
        {
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.58f);
            for (auto& emitter : emitters_)
            {
                emitter->GUI();
            }
            ImGui::PopItemWidth();

            if (ImGui::Button("New Radial"))
            {
                emitters_.push_back(std::make_unique<RadialEmitter>());
            }

            if (ImGui::Button("Enable All"))
            {
                for (auto& emitter : emitters_)
                {
                    emitter->enabled = true;
                }
            }

            ImGui::SameLine();

            if (ImGui::Button("Disable All"))
            {
                for (auto& emitter : emitters_)
                {
                    emitter->enabled = false;
                }
            }
        }
    }
    ImGui::End();

    // Tick emitters
    {
        auto r = std::ranges::remove(emitters_, true, &Emitter::pending_kill);
        emitters_.erase(r.begin(), r.end());

        // Iterate only through emitters that existed before
        for (const size_t emitter_index : std::views::iota(size_t{0}, emitters_.size()))
        {
            auto& emitter = *emitters_[emitter_index];
            emitter.Tick(*this);

            if (emitter.clone_requested)
            {
                emitter.clone_requested = false;
                auto cloned = emitter.Clone();
                cloned->ResetRuntimeState();
                emitters_.push_back(std::move(cloned));
            }
        }
    }

    DrawObjects();

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse)
    {
        const auto mouse_position = GetMousePositionInWorldCoordinates();
        AddObject({
            .old_position = mouse_position,
            .position = mouse_position,
            .color = {1, 0, 0, 1},
            .scale = Vec2f{} + constants::kObjectRadius,
        });
    }

    {
        const Vec2f window_padding{10, 10};

        ImGui::PushFont(big_font_);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ToImVec(window_padding));

        auto text = fmt::format("Objects count: {}", used_objects_count_);
        char* text_begin = text.data();
        char* text_end = std::next(text_begin, static_cast<int>(text.size()));
        const Vec2f text_size = FromImVec(ImGui::CalcTextSize(text_begin, text_end));
        const Vec2f text_window_size = text_size + 2 * window_padding;
        const ImVec2 counter_margin{32.0f, 32.0f};

        constexpr int flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                              ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                              ImGuiWindowFlags_AlwaysAutoResize;

        ImGui::SetNextWindowPos(
            ImVec2{
                viewport->WorkPos.x + viewport->WorkSize.x - counter_margin.x,
                viewport->WorkPos.y + counter_margin.y,
            },
            ImGuiCond_Always,
            ImVec2{1.0f, 0.0f});
        ImGui::SetNextWindowSize(ToImVec(text_window_size));

        if (ImGui::Begin("Counter", nullptr, flags))
        {
            ImGui::TextUnformatted(text_begin, text_end);  // NOLINT
            ImGui::End();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopFont();
    }
}

}  // namespace verlet
