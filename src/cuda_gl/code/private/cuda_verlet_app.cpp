#include "cuda_verlet_app.hpp"

#include "coloring/spawn_color/spawn_color_strategy_rainbow.hpp"
#include "constants.hpp"
#include "cuda_gl_interop.hpp"
#include "emitters/radial_emitter.hpp"
#include "imgui.h"
#include "klgl/error_handling.hpp"
#include "klgl/events/event_listener_method.hpp"
#include "klgl/events/event_manager.hpp"
#include "klgl/template/member_offset.hpp"
#include "klgl/texture/procedural_texture_generator.hpp"

namespace verlet
{
[[nodiscard]] constexpr ImVec2 ToImVec(Vec2f v) noexcept
{
    return ImVec2(v.x(), v.y());
}

[[nodiscard]] constexpr Vec2f FromImVec(ImVec2 v) noexcept
{
    return {v.x, v.y};
}

template <typename T>
[[nodiscard]] constexpr std::span<T> ReinterpretSpan(std::span<uint8_t> span)
{
    klgl::ErrorHandling::Ensure(
        span.size_bytes() % sizeof(T) == 0,
        "Possibly wrong conversion: converting {} bytes to an array of objects with size {}",
        span.size_bytes(),
        sizeof(T));
    return std::span{
        reinterpret_cast<T*>(span.data()),  // NOLINT
        span.size_bytes() / sizeof(T),
    };
}

template <typename... Args>
void CheckResult(cudaError_t result, fmt::format_string<Args...> format_string = "", Args&&... args)
{
    if (result != cudaSuccess)
    {
        std::string message =
            fmt::format("Cuda operation returned an error code. Error: {}. ", cudaGetErrorString(result));
        if (fmt::formatted_size(format_string, args...))
        {
            message.append("\nContext: ");
            fmt::format_to(std::back_inserter(message), format_string, std::forward<Args>(args)...);
        }

        throw cpptrace::runtime_error(std::move(message));
    }
}

template <typename T>
[[nodiscard]] CudaPtr<T> MakeCudaArray(size_t elements_count)
{
    T* device_ptr{};
    cudaMalloc(&device_ptr, sizeof(T) * elements_count);
    return CudaPtr<T>(device_ptr);
}

template <typename T, size_t extent>
void CopyCudaArrayToDevice(std::span<T, extent> host, std::remove_const_t<T>* device)
{
    CheckResult(cudaMemcpy(device, host.data(), host.size_bytes(), cudaMemcpyHostToDevice));
}

template <typename T, size_t extent>
void CopyCudaArrayToDevice(std::span<T, extent> host, const CudaPtr<std::remove_const_t<T>>& device)
{
    CopyCudaArrayToDevice(host, device.get());
}

template <typename T, size_t extent>
void CopyCudaArrayToHost(std::span<T, extent> host, std::remove_const_t<T>* device)
{
    CheckResult(cudaMemcpy(host.data(), device, host.size_bytes(), cudaMemcpyDeviceToHost));
}

template <typename T, size_t extent>
void CopyCudaArrayToHost(std::span<T, extent> host, const CudaPtr<std::remove_const_t<T>>& device)
{
    CopyCudaArrayToHost(host, device.get());
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

    grid_cells_ = MakeCudaArray<GridCell>(constants::kGridNumCells);
    CheckResult(cudaMemset(grid_cells_.get(), 0, constants::kGridNumCells * sizeof(GridCell)));

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

    ColorAttribHelper::EnableVertexAttribArray(a_color_);
    GL::BindBuffer(klgl::GlBufferType::Array, color_vbo_);
    ColorAttribHelper::AttributePointer(a_color_);
    ColorAttribHelper::AttributeDivisor(a_color_, 1);

    PositionAttribHelper::EnableVertexAttribArray(a_position_);
    GL::BindBuffer(klgl::GlBufferType::Array, position_vbo_);
    PositionAttribHelper::AttributePointer(a_position_);
    PositionAttribHelper::AttributeDivisor(a_position_, 1);

    ScaleAttribHelper::EnableVertexAttribArray(a_scale_);
    GL::BindBuffer(klgl::GlBufferType::Array, scale_vbo_);
    ScaleAttribHelper::AttributePointer(a_scale_);
    ScaleAttribHelper::AttributeDivisor(a_scale_, 1);
}

void VerletCudaApp::SpawnPendingObjects()
{
    if (pending_positions_.empty())
    {
        return;
    }

    auto cleanup = [](cudaGraphicsResource* resource)
    {
        if (resource)
        {
            CudaGlInterop::UnmapResource(resource);
            CudaGlInterop::UnregisterBuffer(resource);
        }
    };

    const size_t new_count = used_objects_count_ + pending_positions_.size();

    // Realloc logic
    if (new_count > reserved_objects_count_)
    {
        constexpr size_t growth_step = 3;
        const size_t new_capacity = growth_step * ((new_count + growth_step) / growth_step);

        /* ---------------------------------------- Color ---------------------------------------- */

        auto new_color_vbo = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());
        GL::BindBuffer(klgl::GlBufferType::Array, new_color_vbo);
        GL::BufferData(klgl::GlBufferType::Array, sizeof(Vec4f) * new_capacity, klgl::GlUsage::DynamicDraw);
        auto new_color_resource = CudaGlInterop::RegisterBuffer(new_color_vbo);
        auto new_device_colors = ReinterpretSpan<Vec4f>(CudaGlInterop::MapResourceAndGetPtr(new_color_resource));
        cudaGraphicsResource* prev_color_resource = nullptr;

        if (used_objects_count_ != 0)
        {
            prev_color_resource = CudaGlInterop::RegisterBuffer(color_vbo_);
            auto prev_device_colors = ReinterpretSpan<Vec4f>(CudaGlInterop::MapResourceAndGetPtr(prev_color_resource));
            assert(prev_device_colors.size() == reserved_objects_count_);

            CheckResult(cudaMemcpyAsync(
                new_device_colors.data(),
                prev_device_colors.data(),
                sizeof(Vec4f) * used_objects_count_,
                cudaMemcpyDeviceToDevice,
                cuda_stream_));
        }

        /* ---------------------------------------- Position ---------------------------------------- */

        auto new_position_vbo = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());
        GL::BindBuffer(klgl::GlBufferType::Array, new_position_vbo);
        GL::BufferData(klgl::GlBufferType::Array, sizeof(Vec2f) * new_capacity, klgl::GlUsage::DynamicDraw);
        auto new_position_resource = CudaGlInterop::RegisterBuffer(new_position_vbo);
        auto new_device_positions = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(new_position_resource));
        cudaGraphicsResource* prev_position_resource = nullptr;

        if (used_objects_count_ != 0)
        {
            prev_position_resource = CudaGlInterop::RegisterBuffer(position_vbo_);
            auto prev_device_positions =
                ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(prev_position_resource));
            assert(prev_device_positions.size() == reserved_objects_count_);

            CheckResult(cudaMemcpyAsync(
                new_device_positions.data(),
                prev_device_positions.data(),
                sizeof(Vec2f) * used_objects_count_,
                cudaMemcpyDeviceToDevice,
                cuda_stream_));
        }

        /* ---------------------------------------- Scale ---------------------------------------- */

        auto new_scale_vbo = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());
        GL::BindBuffer(klgl::GlBufferType::Array, new_scale_vbo);
        GL::BufferData(klgl::GlBufferType::Array, sizeof(Vec2f) * new_capacity, klgl::GlUsage::DynamicDraw);
        auto new_scale_resource = CudaGlInterop::RegisterBuffer(new_scale_vbo);
        auto new_device_scales = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(new_scale_resource));
        cudaGraphicsResource* prev_scale_resource = nullptr;

        if (used_objects_count_ != 0)
        {
            prev_scale_resource = CudaGlInterop::RegisterBuffer(scale_vbo_);
            auto prev_device_scales = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(prev_scale_resource));
            assert(prev_device_scales.size() == reserved_objects_count_);

            CheckResult(cudaMemcpyAsync(
                new_device_scales.data(),
                prev_device_scales.data(),
                sizeof(Vec2f) * used_objects_count_,
                cudaMemcpyDeviceToDevice,
                cuda_stream_));
        }

        /* ---------------------------------------- Old Position ---------------------------------------- */
        CudaPtr<Vec2f> new_old_positions;
        {
            void* ptr{};
            CheckResult(cudaMallocAsync(&ptr, sizeof(Vec2f) * new_capacity, cuda_stream_));
            new_old_positions.reset(reinterpret_cast<Vec2f*>(ptr));  // NOLINT
        }
        if (used_objects_count_ != 0)
        {
            CheckResult(cudaMemcpyAsync(
                new_old_positions.get(),
                device_old_positions_.get(),
                sizeof(Vec2f) * used_objects_count_,
                cudaMemcpyDeviceToDevice,
                cuda_stream_));
        }

        color_vbo_ = std::move(new_color_vbo);
        position_vbo_ = std::move(new_position_vbo);
        scale_vbo_ = std::move(new_scale_vbo);
        device_old_positions_ = std::move(new_old_positions);

        CheckResult(cudaStreamSynchronize(cuda_stream_));

        cleanup(new_color_resource);
        cleanup(prev_color_resource);

        cleanup(new_position_resource);
        cleanup(prev_position_resource);

        cleanup(new_scale_resource);
        cleanup(prev_scale_resource);

        // The last step
        RegisterGLBuffers();

        reserved_objects_count_ = new_capacity;
    }

    /* ---------------------------------------- Color ---------------------------------------- */

    auto color_resource = CudaGlInterop::RegisterBuffer(color_vbo_);
    auto device_colors = ReinterpretSpan<Vec4f>(CudaGlInterop::MapResourceAndGetPtr(color_resource));
    assert(device_colors.size() == reserved_objects_count_);

    CheckResult(cudaMemcpyAsync(
        device_colors.data() + used_objects_count_,
        pending_colors_.data(),
        sizeof(Vec4f) * pending_colors_.size(),
        cudaMemcpyHostToDevice,
        cuda_stream_));

    /* ---------------------------------------- Position ---------------------------------------- */

    auto position_resource = CudaGlInterop::RegisterBuffer(position_vbo_);
    auto device_positions = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(position_resource));
    assert(device_positions.size() == reserved_objects_count_);

    CheckResult(cudaMemcpyAsync(
        device_positions.data() + used_objects_count_,
        pending_positions_.data(),
        sizeof(Vec2f) * pending_positions_.size(),
        cudaMemcpyHostToDevice,
        cuda_stream_));

    /* ---------------------------------------- Scale ---------------------------------------- */

    auto scale_resource = CudaGlInterop::RegisterBuffer(scale_vbo_);
    auto device_scale = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(scale_resource));
    assert(device_scale.size() == reserved_objects_count_);

    CheckResult(cudaMemcpyAsync(
        device_scale.data() + used_objects_count_,
        pending_scales_.data(),
        sizeof(Vec2f) * pending_scales_.size(),
        cudaMemcpyHostToDevice,
        cuda_stream_));

    /* ---------------------------------------- Old Position ---------------------------------------- */

    CheckResult(cudaMemcpyAsync(
        device_old_positions_.get() + used_objects_count_,
        pending_old_positions_.data(),
        sizeof(Vec2f) * pending_old_positions_.size(),
        cudaMemcpyHostToDevice,
        cuda_stream_));

    CheckResult(cudaStreamSynchronize(cuda_stream_));

    cleanup(color_resource);
    cleanup(scale_resource);
    cleanup(position_resource);
    used_objects_count_ = new_count;

    pending_colors_.clear();
    pending_positions_.clear();
    pending_old_positions_.clear();
    pending_scales_.clear();

    // auto print_buf =
    //     [this,
    //      &cleanup]<typename T>(std::tuple<T>, std::string_view title, const klgl::GlObject<klgl::GlBufferId>&
    //      vbo)
    // {
    //     auto resource = CudaGlInterop::RegisterBuffer(vbo);
    //     auto device_data =
    //         ReinterpretSpan<T>(CudaGlInterop::MapResourceAndGetPtr(resource)).subspan(0, used_objects_count_);
    //     std::vector<T> host_data(device_data.size());
    //     CheckResult(cudaMemcpy(
    //         host_data.data(),
    //         device_data.data(),
    //         sizeof(T) * device_data.size(),
    //         cudaMemcpyDeviceToHost));

    //     fmt::print("{}: ", title);
    //     for (auto& v : host_data)
    //     {
    //         fmt::print("{{");
    //         fmt::print("{}", v[0]);
    //         for (size_t i = 1; i != T::Size(); ++i)
    //         {
    //             fmt::print(", {}", v[i]);
    //         }
    //         fmt::print("}}, ");
    //     }

    //     fmt::println("");

    //     cleanup(resource);
    // };
    // print_buf(std::tuple<Vec2f>{}, "position", position_vbo_);
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

            // Take pointer to positions from VBO
            auto positions_resource = CudaGlInterop::RegisterBuffer(position_vbo_);
            auto device_positions = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(positions_resource));
            klgl::ErrorHandling::Ensure(
                device_positions.size() == reserved_objects_count_,
                "Unexpected size of positions array on device. Expected: {}. Actual: {}",
                reserved_objects_count_,
                device_positions.size());

            // Now it is a view to an array of full capacity. We need only subset of it
            device_positions = device_positions.subspan(0, used_objects_count_);

            for (size_t substep = 0; substep != constants::kNumSubSteps; ++substep)
            {
                CheckResult(
                    cudaMemsetAsync(grid_cells_.get(), 0, sizeof(GridCell) * constants::kGridNumCells, cuda_stream_));
                Kernels::PopulateGrid(cuda_stream_, grid_cells_.get(), used_objects_count_, device_positions.data());

                // CheckGrid(device_positions);

                for (size_t offset_y = 0; offset_y != 3; ++offset_y)
                {
                    for (size_t offset_x = 0; offset_x != 3; ++offset_x)
                    {
                        // fmt::println("Substep begin. Grid size: {}x{}. Offset: {} {}\n",
                        // constants::kGridSize.x(), constants::kGridSize.y(), offset_x, offset_y);
                        Kernels::SolveCollisions_ManyRows(
                            cuda_stream_,
                            grid_cells_.get(),
                            device_positions.data(),
                            {offset_x, offset_y});
                        // cudaStreamSynchronize(cuda_stream_);
                        // fmt::println("Substep end\n");
                    }
                }
                Kernels::UpdatePositions(
                    cuda_stream_,
                    used_objects_count_,
                    device_positions.data(),
                    device_old_positions_.get());
            }

            cudaStreamSynchronize(cuda_stream_);
            CudaGlInterop::UnmapResource(positions_resource);
            CudaGlInterop::UnregisterBuffer(positions_resource);
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

void VerletCudaApp::CheckGrid(std::span<PositionType> device_positions)
{
    static std::vector<GridCell> cells;
    cells.resize(constants::kGridNumCells);
    CheckResult(cudaMemcpyAsync(
        cells.data(),
        grid_cells_.get(),
        std::span{cells}.size_bytes(),
        cudaMemcpyDeviceToHost,
        cuda_stream_));
    std::vector<Vec2f> positions;
    positions.resize(used_objects_count_);
    CheckResult(cudaMemcpyAsync(
        positions.data(),
        device_positions.data(),
        device_positions.size_bytes(),
        cudaMemcpyDeviceToHost,
        cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);

    for (size_t cell_index = 0; cell_index != 100; ++cell_index)
    {
        const GridCell& cell = cells[cell_index];
        klgl::ErrorHandling::Ensure(
            cell.num_objects <= constants::kGridMaxObjectsInCell,
            "Cell {} contains more objects ({}) than it supposed to ({}).",
            cell_index,
            cell.num_objects,
            constants::kGridMaxObjectsInCell);
        for (size_t i = 0; i != cell.num_objects; ++i)
        {
            auto& object_index = cell.objects[i];
            const Vec2f& object_pos = positions[object_index];
            const auto expected_cell_index = DeviceGrid::LocationToCellIndex(object_pos);
            klgl::ErrorHandling::Ensure(
                cell_index == expected_cell_index,
                "An object {} with position ({}, {}) is in the cell {}. It is expected to be in the cell {}",
                object_index,
                object_pos.x(),
                object_pos.y(),
                cell_index,
                expected_cell_index);
        }
    }

    size_t objects_without_cell = 0;
    for (size_t object_index = 0; object_index != positions.size(); ++object_index)
    {
        const auto cell_index = DeviceGrid::LocationToCellIndex(positions[object_index]);
        const GridCell& cell = cells[cell_index];

        bool object_in_cell = false;
        for (size_t i = 0; i != cell.num_objects; ++i)
        {
            if (cell.objects[i] == object_index)
            {
                object_in_cell = true;
                break;
            }
        }

        objects_without_cell += object_in_cell ? 0 : 1;
        klgl::ErrorHandling::Ensure(
            object_in_cell || cell.num_objects == constants::kGridMaxObjectsInCell,
            "The object {} belongs to the cell {} but not registered there. The cell is also not full.",
            object_index,
            cell_index);
    }

    if (objects_without_cell)
    {
        fmt::println("Objects without cell: {}", objects_without_cell);
    }
}

void VerletCudaApp::AddObject(const VerletObject& object)
{
    pending_old_positions_.push_back(object.old_position);
    pending_positions_.push_back(object.position);
    pending_colors_.push_back(object.color);
    pending_scales_.push_back(object.scale);
}

}  // namespace verlet
