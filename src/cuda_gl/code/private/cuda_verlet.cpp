#include <EverydayTools/Math/Math.hpp>
#include <random>

#include "klgl/application.hpp"
#include "klgl/error_handling.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/opengl/vertex_attribute_helper.hpp"
#include "klgl/reflection/matrix_reflect.hpp"  // IWYU pragma: keep
#include "klgl/shader/shader.hpp"
#include "klgl/template/member_offset.hpp"
#include "klgl/ui/imgui_helpers.hpp"
#include "klgl/window.hpp"
#include "klgl/texture/procedural_texture_generator.hpp"
#include "klgl/texture/texture.hpp"
#include <klgl/mesh/procedural_mesh_generator.hpp>
#include <klgl/mesh/mesh_data.hpp>
#include "cuda/gl_interop.hpp"
#include <EverydayTools/Math/FloatRange.hpp>
#include "camera.hpp"
#include "klgl/events/event_listener_method.hpp"
#include "klgl/events/event_manager.hpp"
#include "klgl/events/mouse_events.hpp"
#include "cuda/cuda_helpers.hpp"
#include "kernels.hpp"
#include "constants.hpp"
#include "imgui.h"
#include "time.hpp"
#include <fmt/chrono.h>

namespace verlet_cuda
{

struct MeshVertex
{
    Vec2f vertex;
    Vec2f tex_coord;
};

class VerletCudaApp : public klgl::Application
{
public:
    using GL = klgl::OpenGl;
    using ColorType = edt::Vec4f;
    using ColorAttribHelper = klgl::VertexBufferHelperStatic<ColorType>;
    using PositionType = edt::Vec2f;
    using PositionAttribHelper = klgl::VertexBufferHelperStatic<PositionType>;
    using ScaleType = edt::Vec2f;
    using ScaleAttribHelper = klgl::VertexBufferHelperStatic<ScaleType>;

    static constexpr unsigned kSeed = 12345;

    VerletCudaApp()
    {
        event_listener_ = klgl::events::EventListenerMethodCallbacks<&VerletCudaApp::OnMouseScroll>::CreatePtr(this);
        GetEventManager().AddEventListener(*event_listener_);
    }

    void Initialize() override
    {

        klgl::Application::Initialize();

        GL::SetClearColor({});
        GetWindow().SetSize(1000, 1000);
        GetWindow().SetTitle("Cuda and OpenGL");

        GL::EnableBlending();
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        shader_ = std::make_unique<klgl::Shader>("cuda_verlet");
        shader_->Use();

        CreateMesh();
        CreateCircleMaskTexture();
        GenerateRandomObjects();

        const auto& shader_info = shader_->GetInfo();

        a_color_ = shader_info.VerifyAndGetVertexAttributeLocation<ColorType>("a_color");
        color_vbo_ = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());

        a_position_ = shader_info.VerifyAndGetVertexAttributeLocation<PositionType>("a_position");
        position_vbo_ = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());

        a_scale_ = shader_info.VerifyAndGetVertexAttributeLocation<ScaleType>("a_scale");
        scale_vbo_ = klgl::GlObject<klgl::GlBufferId>::CreateFrom(GL::GenBuffer());

        ColorAttribHelper::EnableVertexAttribArray(a_color_);
        GL::BindBuffer(klgl::GlBufferType::Array, color_vbo_);
        ColorAttribHelper::AttributePointer(a_color_);
        ColorAttribHelper::AttributeDivisor(a_color_, 1);
        GL::BufferData(klgl::GlBufferType::Array, sizeof(ColorType) * constants::kMaxObjectsCount, klgl::GlUsage::DynamicDraw);

        PositionAttribHelper::EnableVertexAttribArray(a_position_);
        GL::BindBuffer(klgl::GlBufferType::Array, position_vbo_);
        PositionAttribHelper::AttributePointer(a_position_);
        PositionAttribHelper::AttributeDivisor(a_position_, 1);
        GL::BufferData(klgl::GlBufferType::Array, sizeof(PositionType) * constants::kMaxObjectsCount, klgl::GlUsage::DynamicDraw);

        ScaleAttribHelper::EnableVertexAttribArray(a_scale_);
        GL::BindBuffer(klgl::GlBufferType::Array, scale_vbo_);
        ScaleAttribHelper::AttributePointer(a_scale_);
        ScaleAttribHelper::AttributeDivisor(a_scale_, 1);
        GL::BufferData(klgl::GlBufferType::Array, sizeof(ScaleType) * constants::kMaxObjectsCount, klgl::GlUsage::DynamicDraw);

        cudaStreamCreate(&cuda_stream_);

        positions_vbo_cuda_ = CudaGlInterop::RegisterBuffer(position_vbo_);

        grid_cells_ = MakeCudaArray<GridCell>(constants::kGridNumCells);
        cudaMemset(grid_cells_.get(), 0, constants::kGridNumCells * sizeof(GridCell));
        device_old_positions_ = MakeCudaArray<PositionType>(constants::kMaxObjectsCount);
        CopyCudaArrayToDevice(std::span{positions_}, device_old_positions_);

        SendObjects(0);
    }

    void SendObjects(size_t ignore_first_n)
    {
        GL::BindBuffer(klgl::GlBufferType::Array, position_vbo_);
        GL::BufferSubData(klgl::GlBufferType::Array, ignore_first_n, std::span{ positions_ }.subspan(ignore_first_n));
        GL::BindBuffer(klgl::GlBufferType::Array, color_vbo_);
        GL::BufferSubData(klgl::GlBufferType::Array, ignore_first_n, std::span{ colors_ }.subspan(ignore_first_n));
        GL::BindBuffer(klgl::GlBufferType::Array, scale_vbo_);
        GL::BufferSubData(klgl::GlBufferType::Array, ignore_first_n, std::span{ scales_ }.subspan(ignore_first_n));

        CopyCudaArrayToDevice(std::span{ positions_ }.subspan(ignore_first_n), device_old_positions_.get() + ignore_first_n);

        CudaGlInterop::UnregisterBuffer(positions_vbo_cuda_);
        positions_vbo_cuda_ = CudaGlInterop::RegisterBuffer(position_vbo_);
    }

    void GenerateRandomObjects()
    {
        std::mt19937 rnd(kSeed); // NOLINT
        std::uniform_real_distribution<float> pos_distr_x(constants::kWorldRange.x.begin + 10.f, constants::kWorldRange.x.end - 10.f);
        std::uniform_real_distribution<float> pos_distr_y(constants::kWorldRange.y.begin + 10.f, constants::kWorldRange.y.end - 10.f);
        std::uniform_real_distribution<float> color_distr(0.f, 1.f);
        size_t objects_count = constants::kMaxObjectsCount;
        colors_.reserve(objects_count);
        scales_.reserve(objects_count);
        positions_.reserve(objects_count);
        for ([[maybe_unused]] size_t i : std::views::iota(size_t{ 0 }, objects_count))
        {
            AddObject(
                { pos_distr_x(rnd), pos_distr_y(rnd) },
                { color_distr(rnd), color_distr(rnd), color_distr(rnd), color_distr(rnd) },
                Vec2f{} + 0.5f);
        }

        // AddObject(
        //    Vec2f{0.01f, 0},
        //    { 1, 0, 0, 1 },
        //    Vec2f{} + 0.5f);
    }

    void CreateMesh()
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

        mesh_ = klgl::MeshOpenGL::MakeFromData(std::span{ vertices }, std::span{ mesh_data.indices }, mesh_data.topology);
        mesh_->Bind();

        a_vertex_ = shader_->GetInfo().VerifyAndGetVertexAttributeLocation<edt::Vec2f>("a_vertex");
        {
            using AttribHelper = klgl::VertexBufferHelperStatic<edt::Vec2f>;
            AttribHelper::EnableVertexAttribArray(a_vertex_);
            AttribHelper::AttributePointer(a_vertex_, sizeof(MeshVertex), klgl::MemberOffset<&MeshVertex::vertex>());
            AttribHelper::AttributeDivisor(a_vertex_, 0);
        }

        a_tex_coord_ = shader_->GetInfo().VerifyAndGetVertexAttributeLocation<edt::Vec2f>("a_tex_coord");
        {
            using AttribHelper = klgl::VertexBufferHelperStatic<edt::Vec2f>;
            AttribHelper::EnableVertexAttribArray(a_tex_coord_);
            AttribHelper::AttributePointer(a_tex_coord_, sizeof(MeshVertex), klgl::MemberOffset<&MeshVertex::tex_coord>());
            AttribHelper::AttributeDivisor(a_tex_coord_, 0);
        }
    }

    void CreateCircleMaskTexture()
    {
        constexpr auto size = edt::Vec2<size_t>{} + 128;
        texture_ = klgl::Texture::CreateEmpty(size, klgl::GlTextureInternalFormat::R8);
        const auto pixels = klgl::ProceduralTextureGenerator::CircleMask(size, 2);
        texture_->SetPixels<klgl::GlPixelBufferLayout::R>(std::span{ pixels });
        klgl::OpenGl::SetTextureMinFilter(klgl::GlTargetTextureType::Texture2d, klgl::GlTextureFilter::Nearest);
        klgl::OpenGl::SetTextureMagFilter(klgl::GlTargetTextureType::Texture2d, klgl::GlTextureFilter::Linear);
        glGenerateMipmap(GL_TEXTURE_2D);
        shader_->SetUniform(u_texture_, *texture_);
    }

    void AddObject(const edt::Vec2f& position, const edt::Vec4f& color, const edt::Vec2f& scale)
    {
        positions_.push_back(position);
        colors_.push_back(color);
        scales_.push_back(scale);
    }

    void UpdateCamera()
    {
        camera_.Update(constants::kWorldRange);

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
    
    void OnMouseScroll(const klgl::events::OnMouseScroll& event)
    {
        if (!ImGui::GetIO().WantCaptureMouse)
        {
            camera_.Zoom(event.value.y() * camera_.zoom_speed);
        }
    }

    void UpdateRenderTransforms()
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

    Vec2f GetMousePositionInWorldCoordinates() const
    {
        const auto screen_range = edt::FloatRange2Df::FromMinMax({}, GetWindow().GetSize2f());  // 0 -> resolution
        auto [x, y] = ImGui::GetMousePos();
        y = screen_range.y.Extent() - y;
        return edt::Math::TransformPos(screen_to_world_, Vec2f{ x, y });
    }

    void Tick() override
    {
        klgl::Application::Tick();

        UpdateCamera();
        UpdateRenderTransforms();

        // Take pointer to positions from VBO
        auto device_positions = ReinterpretSpan<Vec2f>(CudaGlInterop::MapResourceAndGetPtr(positions_vbo_cuda_));
        assert(device_positions.size() == constants::kMaxObjectsCount);

        // Now it is a view to an array of full capacity. We need only subset of it
        device_positions = device_positions.subspan(0, positions_.size());

        // Do simulation substeps
        auto sim_time = MeasureTime([&] {
            for (size_t substep = 0; substep != constants::kNumSubSteps; ++substep) {
                cudaMemsetAsync(grid_cells_.get(), 0, sizeof(GridCell) * constants::kGridNumCells, cuda_stream_);
                Kernels::PopulateGrid(cuda_stream_, grid_cells_.get(), positions_.size(), device_positions.data());

                // CheckGrid(device_positions);


                constexpr bool kDoRows = false;
                if constexpr (kDoRows) {
                    for (size_t cell_y = 1; cell_y != constants::kGridSize.y() - 1; ++cell_y) {
                        for (size_t offset_x = 0; offset_x != 3; ++offset_x) {
                            Kernels::SolveCollisions_OneRow(cuda_stream_, grid_cells_.get(), device_positions.data(), cell_y, offset_x);
                        }
                    }
                } else {
                    for (size_t offset_y = 0; offset_y != 3; ++offset_y) {
                        for (size_t offset_x = 0; offset_x != 3; ++offset_x) {
                            // fmt::println("Substep begin. Grid size: {}x{}. Offset: {} {}\n", constants::kGridSize.x(), constants::kGridSize.y(), offset_x, offset_y);
                            Kernels::SolveCollisions_ManyRows(cuda_stream_, grid_cells_.get(), device_positions.data(), {offset_x, offset_y});
                            // cudaStreamSynchronize(cuda_stream_);
                            // fmt::println("Substep end\n");
                        }
                    }
                }
                Kernels::UpdatePositions(cuda_stream_, positions_.size(), device_positions.data(), device_old_positions_.get());
            }

            cudaStreamSynchronize(cuda_stream_);
            CudaGlInterop::UnmapResource(positions_vbo_cuda_);
        });

        if (ImGui::CollapsingHeader("Perf")) {
            std::string txt = fmt::format("{}", sim_time);
            ImGui::Text("%s", txt.data());
            ImGui::Text("Framerate: %f", GetFramerate());
        }


        shader_->SetUniform(u_world_to_view_, world_to_view_.Transposed());
        shader_->SendUniforms();

        mesh_->DrawInstanced(positions_.size());

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse)
        {
            const auto mouse_position = GetMousePositionInWorldCoordinates();
            size_t prev_count = positions_.size();
            AddObject(mouse_position, { 1, 0, 0, 1 }, { 0.5f, 0.5f });
            SendObjects(prev_count);
        }
    }

    void CheckGrid(std::span<PositionType> device_positions)
    {
        static std::vector<GridCell> cells;
        cells.resize(constants::kGridNumCells);
        auto err = cudaMemcpyAsync(cells.data(), grid_cells_.get(), std::span{cells}.size_bytes(), cudaMemcpyDeviceToHost, cuda_stream_);
        assert(err == cudaSuccess);
        err = cudaMemcpyAsync(positions_.data(), device_positions.data(), device_positions.size_bytes(), cudaMemcpyDeviceToHost, cuda_stream_);
        assert(err == cudaSuccess);
        cudaStreamSynchronize(cuda_stream_);

        for (size_t cell_index = 0; cell_index != 100; ++cell_index)
        {
            const GridCell& cell = cells[cell_index];
            assert(cell.num_objects <= constants::kGridMaxObjectsInCell);
            for (size_t i = 0; i != cell.num_objects; ++i)
            {
                auto& object_index = cell.objects[i];
                const auto expected_cell_index = DeviceGrid::LocationToCellIndex(positions_[object_index]);
                assert(cell_index == expected_cell_index);
            }
        }

        size_t objects_without_cell = 0;
        for (size_t object_index = 0; object_index != positions_.size(); ++object_index)
        {
            const auto cell_index = DeviceGrid::LocationToCellIndex(positions_[object_index]);
            const GridCell& cell = cells[cell_index];

            assert(cell.num_objects <= constants::kGridMaxObjectsInCell);
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
            assert(object_in_cell || cell.num_objects == constants::kGridMaxObjectsInCell);
        }

        if (objects_without_cell)
        {
            fmt::println("Objects without cell: {}", objects_without_cell);
        }
    }

    cudaStream_t cuda_stream_;

    std::unique_ptr<klgl::events::IEventListener> event_listener_;

    Camera camera_{};
    std::shared_ptr<klgl::Shader> shader_;

    size_t a_vertex_{};
    size_t a_tex_coord_{};

    size_t a_color_{};
    klgl::GlObject<klgl::GlBufferId> color_vbo_;
    std::vector<ColorType> colors_;

    size_t a_position_{};
    klgl::GlObject<klgl::GlBufferId> position_vbo_;
    std::vector<PositionType> positions_;
    cudaGraphicsResource* positions_vbo_cuda_ = nullptr;

    CudaPtr<PositionType> device_old_positions_;

    size_t a_scale_{};
    klgl::GlObject<klgl::GlBufferId> scale_vbo_;
    std::vector<ScaleType> scales_;

    std::shared_ptr<klgl::MeshOpenGL> mesh_;

    klgl::UniformHandle u_texture_{ "u_texture" };
    std::unique_ptr<klgl::Texture> texture_;

    klgl::UniformHandle u_world_to_view_ { "u_world_to_view" };
    Mat3f world_to_view_ = edt::Mat3f::Identity();

    Mat3f world_to_camera_ = edt::Mat3f::Identity();
    Mat3f screen_to_world_ = edt::Mat3f::Identity();

    CudaPtr<GridCell> grid_cells_;
};

void Main()
{
    VerletCudaApp app;
    app.Run();
}

} //

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(verlet_cuda::Main);
    return 0;
}
