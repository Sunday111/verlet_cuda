#include "verlet_cuda_app.hpp"

#include "coloring/spawn_color/spawn_color_strategy_rainbow.hpp"
#include "constants.hpp"
#include "emitters/radial_emitter.hpp"
#include "imgui.h"
#include "klgl/error_handling.hpp"
#include "klgl/events/event_listener_method.hpp"
#include "klgl/events/event_manager.hpp"
#include "klgl/reflection/matrix_reflect.hpp"  // IWYU pragma: keep
#include "klgl/template/member_offset.hpp"
#include "klgl/texture/procedural_texture_generator.hpp"

namespace verlet
{
[[nodiscard]] static constexpr ImVec2 ToImVec(Vec2f v) noexcept
{
    return ImVec2(v.x(), v.y());
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

struct MeshVertex
{
    Vec2f vertex;
    Vec2f tex_coord;
};

VerletCudaApp::VerletCudaApp()
{
    event_listener_ = klgl::events::EventListenerMethodCallbacks<&VerletCudaApp::OnMouseScroll>::CreatePtr(this);
    GetEventManager().AddEventListener(*event_listener_);
    spawn_color_strategy_ = std::make_unique<SpawnColorStrategyRainbow>(*this);
}

VerletCudaApp::~VerletCudaApp() = default;

void VerletCudaApp::Initialize()
{
    klgl::Application::Initialize();

    SetTargetFramerate({60.f});

    GL::SetClearColor({});
    GetWindow().SetSize(1920, 1080);
    GetWindow().SetTitle("Cuda and OpenGL");

    GL::EnableBlending();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shader_ = std::make_unique<klgl::Shader>("cuda_verlet");
    shader_->Use();

    CreateMesh();
    CreateCircleMaskTexture();

    const auto& shader_info = shader_->GetInfo();
    a_color_ = shader_info.VerifyAndGetVertexAttributeLocation<ColorType>("a_color");
    a_position_ = shader_info.VerifyAndGetVertexAttributeLocation<PositionType>("a_position");
    a_scale_ = shader_info.VerifyAndGetVertexAttributeLocation<ScaleType>("a_scale");

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
}

void VerletCudaApp::RegisterGLBuffers()
{
    mesh_->Bind();

    GL::BindBuffer(klgl::GlBufferType::Array, objects_vbo_);

    ColorAttribHelper::EnableVertexAttribArray(a_color_);
    ColorAttribHelper::AttributePointer(a_color_, sizeof(VerletObject), klgl::MemberOffset<&VerletObject::color>());
    ColorAttribHelper::AttributeDivisor(a_color_, 1);

    PositionAttribHelper::EnableVertexAttribArray(a_position_);
    PositionAttribHelper::AttributePointer(
        a_position_,
        sizeof(VerletObject),
        klgl::MemberOffset<&VerletObject::position>());
    PositionAttribHelper::AttributeDivisor(a_position_, 1);

    ScaleAttribHelper::EnableVertexAttribArray(a_scale_);
    ScaleAttribHelper::AttributePointer(a_scale_, sizeof(VerletObject), klgl::MemberOffset<&VerletObject::scale>());
    ScaleAttribHelper::AttributeDivisor(a_scale_, 1);
}

std::tuple<CudaMappedGraphicsResourcePtr, std::span<VerletObject>> VerletCudaApp::ReserveAndGetDevicePtr(
    size_t required_size)
{
    if (required_size <= reserved_objects_count_)
    {
        return CudaGlInterop::MapAndGet<VerletObject>(objects_vbo_resource_);
    }

    constexpr size_t kCapacityGrowthStep = 5000;
    const size_t new_capacity = kCapacityGrowthStep * ((required_size + kCapacityGrowthStep) / kCapacityGrowthStep);
    auto new_vbo = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());
    GL::BindBuffer(klgl::GlBufferType::Array, new_vbo);
    GL::BufferData(klgl::GlBufferType::Array, sizeof(VerletObject) * new_capacity, klgl::GlUsage::DynamicDraw);
    CudaGraphicsResourcePtr new_resource(CudaGlInterop::RegisterResource(new_vbo));

    auto [mapped_new_resource, new_device_objects] = CudaGlInterop::MapAndGet<VerletObject>(new_resource);

    if (used_objects_count_ != 0)
    {
        auto [mapped_prev_resource, prev_device_objects] =
            CudaGlInterop::MapAndGet<VerletObject>(objects_vbo_resource_);
        klgl::ErrorHandling::Ensure(
            prev_device_objects.size() == reserved_objects_count_,
            "Unexpected size of device array. Expected: {}. Actual: {}",
            reserved_objects_count_,
            prev_device_objects.size());
        CudaMemcpy(
            prev_device_objects.subspan(0, used_objects_count_),
            new_device_objects.subspan(0, used_objects_count_),
            cudaMemcpyDeviceToDevice,
            cuda_stream_);
    }

    objects_vbo_ = std::move(new_vbo);
    objects_vbo_resource_ = std::move(new_resource);
    RegisterGLBuffers();

    reserved_objects_count_ = new_capacity;
    return {std::move(mapped_new_resource), new_device_objects};
}

void VerletCudaApp::SpawnPendingObjects()
{
    if (pending_objects_.empty())
    {
        return;
    }

    const size_t new_count = used_objects_count_ + pending_objects_.size();
    auto [mapped_resource, device_objects] = ReserveAndGetDevicePtr(new_count);
    klgl::ErrorHandling::Ensure(
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

void VerletCudaApp::CreateMesh()
{
    // Create quad mesh
    const auto mesh_data = klgl::ProceduralMeshGenerator::GenerateQuadMesh();

    std::vector<MeshVertex> vertices;
    vertices.reserve(mesh_data.vertices.size());
    for (size_t i = 0; i != mesh_data.vertices.size(); ++i)
    {
        vertices.emplace_back(MeshVertex{
            .vertex = mesh_data.vertices[i],
            .tex_coord = mesh_data.texture_coordinates[i],
        });
    }

    mesh_ = klgl::MeshOpenGL::MakeFromData(std::span{vertices}, std::span{mesh_data.indices}, mesh_data.topology);
    mesh_->Bind();

    a_vertex_ = shader_->GetInfo().VerifyAndGetVertexAttributeLocation<Vec2f>("a_vertex");
    {
        using AttribHelper = klgl::VertexBufferHelperStatic<Vec2f>;
        AttribHelper::EnableVertexAttribArray(a_vertex_);
        AttribHelper::AttributePointer(a_vertex_, sizeof(MeshVertex), klgl::MemberOffset<&MeshVertex::vertex>());
        AttribHelper::AttributeDivisor(a_vertex_, 0);
    }

    a_tex_coord_ = shader_->GetInfo().VerifyAndGetVertexAttributeLocation<Vec2f>("a_tex_coord");
    {
        using AttribHelper = klgl::VertexBufferHelperStatic<Vec2f>;
        AttribHelper::EnableVertexAttribArray(a_tex_coord_);
        AttribHelper::AttributePointer(a_tex_coord_, sizeof(MeshVertex), klgl::MemberOffset<&MeshVertex::tex_coord>());
        AttribHelper::AttributeDivisor(a_tex_coord_, 0);
    }
}

void VerletCudaApp::CreateCircleMaskTexture()
{
    constexpr auto size = Vec2<size_t>{} + 128;
    texture_ = klgl::Texture::CreateEmpty(size, klgl::GlTextureInternalFormat::R8);
    const auto pixels = klgl::ProceduralTextureGenerator::CircleMask(size, 2);
    texture_->SetPixels<klgl::GlPixelBufferLayout::R>(std::span{pixels});
    klgl::OpenGl::SetTextureMinFilter(klgl::GlTargetTextureType::Texture2d, klgl::GlTextureFilter::Nearest);
    klgl::OpenGl::SetTextureMagFilter(klgl::GlTargetTextureType::Texture2d, klgl::GlTextureFilter::Linear);
    glGenerateMipmap(GL_TEXTURE_2D);
    shader_->SetUniform(u_texture_, *texture_);
}

void VerletCudaApp::UpdateCamera()
{
    camera_.Update(constants::kWorldRange, GetWindow().GetSize2f());

    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        Vec2f offset{};
        if (ImGui::IsKeyDown(ImGuiKey_W)) offset.y() += 1.f;
        if (ImGui::IsKeyDown(ImGuiKey_S)) offset.y() -= 1.f;
        if (ImGui::IsKeyDown(ImGuiKey_D)) offset.x() += 1.f;
        if (ImGui::IsKeyDown(ImGuiKey_A)) offset.x() -= 1.f;

        camera_.Pan((GetLastFrameDurationSeconds() * camera_.GetRange().Extent() * offset) * camera_.pan_speed);
    }
}

void VerletCudaApp::OnMouseScroll(const klgl::events::OnMouseScroll& event)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        camera_.Zoom(event.value.y() * camera_.zoom_speed);
    }
}

void VerletCudaApp::UpdateRenderTransforms()
{
    const auto screen_range = edt::FloatRange2Df::FromMinMax({}, GetWindow().GetSize2f());
    const auto view_range = edt::FloatRange2Df::FromMinMax(Vec2f{} - 1, Vec2f{} + 1);
    const auto camera_to_world_vector = constants::kWorldRange.Uniform(.5f) - camera_.GetEye();
    const auto camera_extent = camera_.GetRange().Extent();

    world_to_camera_ = edt::Math::TranslationMatrix(camera_to_world_vector);
    auto camera_to_view_ = edt::Math::ScaleMatrix(view_range.Extent() / camera_extent);
    world_to_view_ = camera_to_view_.MatMul(world_to_camera_);

    const auto screen_to_view =
        edt::Math::TranslationMatrix(Vec2f{} - 1).MatMul(edt::Math::ScaleMatrix(2 / screen_range.Extent()));
    const auto view_to_camera = edt::Math::ScaleMatrix(camera_extent / view_range.Extent());
    const auto camera_to_world = edt::Math::TranslationMatrix(-camera_to_world_vector);
    screen_to_world_ = camera_to_world.MatMul(view_to_camera).MatMul(screen_to_view);
}

Vec2f VerletCudaApp::GetMousePositionInWorldCoordinates() const
{
    const auto screen_range = edt::FloatRange2Df::FromMinMax({}, GetWindow().GetSize2f());  // 0 -> resolution
    auto [x, y] = ImGui::GetMousePos();
    y = screen_range.y.Extent() - y;
    return edt::Math::TransformPos(screen_to_world_, Vec2f{x, y});
}

void VerletCudaApp::Tick()
{
    klgl::Application::Tick();

    UpdateCamera();
    UpdateRenderTransforms();

    // Do simulation substeps
    auto sim_time = MeasureTime(
        [&]
        {
            if (used_objects_count_ == 0) return;

            auto [mapped_resource, device_objects] = CudaGlInterop::MapAndGet<VerletObject>(objects_vbo_resource_);
            klgl::ErrorHandling::Ensure(
                device_objects.size() == reserved_objects_count_,
                "Unexpected size of positions array on device. Expected: {}. Actual: {}",
                reserved_objects_count_,
                device_objects.size());

            // Now it is a view to an array of full capacity. We need only subset of it
            device_objects = device_objects.subspan(0, used_objects_count_);

            for (size_t substep = 0; substep != constants::kNumSubSteps; ++substep)
            {
                CheckResult(
                    cudaMemsetAsync(grid_cells_.get(), 0, sizeof(GridCell) * constants::kGridNumCells, cuda_stream_));
                Kernels::PopulateGrid(cuda_stream_, grid_cells_.get(), used_objects_count_, device_objects.data());

                // CheckGrid(device_positions);

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

    if (ImGui::CollapsingHeader("Perf"))
    {
        std::string txt = fmt::format("{}", sim_time);
        ImGui::Text("%s", txt.data());                 // NOLINT
        ImGui::Text("Framerate: %f", GetFramerate());  // NOLINT
    }

    if (ImGui::CollapsingHeader("Emitters"))
    {
        for (auto& emitter : emitters_)
        {
            emitter->GUI();
        }

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

    shader_->SetUniform(u_world_to_view_, world_to_view_.Transposed());
    shader_->SendUniforms();

    mesh_->DrawInstanced(used_objects_count_);

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
        const Vec2f root_window_size = GetWindow().GetSize2f();
        const Vec2f text_center{(root_window_size.x() - text_size.x()) / 2, 150};
        const Vec2f text_window_size = text_size + 2 * window_padding;

        constexpr int flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                              ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;

        ImGui::SetNextWindowPos(ToImVec(text_center));
        ImGui::SetNextWindowSize(ToImVec(text_window_size));

        if (ImGui::Begin("Counter", nullptr, flags))
        {
            // ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (text_window_size.x - text_size.x) / 2);

            ImGui::TextUnformatted(text_begin, text_end);  // NOLINT
            ImGui::End();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopFont();
    }

    SpawnPendingObjects();
}

}  // namespace verlet
