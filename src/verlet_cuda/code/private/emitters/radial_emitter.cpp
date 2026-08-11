#include "radial_emitter.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

#include "coloring/spawn_color/spawn_color_strategy.hpp"
#include "constants.hpp"
#include "edt/math/math.hpp"
#include "klvk/ui/imgui_helpers.hpp"
#include "verlet_cuda_app.hpp"

namespace verlet
{
namespace
{
constexpr size_t kMaxSpawnPoints = 4096;
constexpr float kMaxSpeed = 240.f;

[[nodiscard]] size_t ClampSpawnPointCount(float estimate)
{
    if (!(estimate > 1.f)) return 1;
    if (estimate >= static_cast<float>(kMaxSpawnPoints)) return kMaxSpawnPoints;
    return static_cast<size_t>(estimate);
}

[[nodiscard]] float NormalizeDegrees(float degrees)
{
    if (!edt::Math::IsFinite(degrees)) return 0.f;
    return std::remainder(degrees, 360.f);
}
}  // namespace

RadialEmitter::RadialEmitter(const RadialEmitterConfig& in_config) : config(in_config)
{
    state = {.phase_degrees = NormalizeDegrees(in_config.phase_degrees)};
}

void RadialEmitter::Tick(VerletCudaApp& app)
{
    if (!enabled) return;
    const size_t remaining_object_capacity = app.GetRemainingObjectCapacity();
    if (remaining_object_capacity == 0) return;

    if (!edt::Math::IsFinite(config.position) || !edt::Math::IsFinite(config.radius) || config.radius < 0.f ||
        !edt::Math::IsFinite(config.sector_degrees) || !edt::Math::IsFinite(config.speed_factor))
    {
        return;
    }

    const float sector_radians = edt::Math::DegToRad(std::clamp(config.sector_degrees, 0.f, 360.f));
    const size_t num_directions = std::min(
        remaining_object_capacity,
        ClampSpawnPointCount(
            sector_radians * (config.radius + constants::kObjectRadius) / (2 * constants::kObjectRadius)));
    const float phase_radians = sector_radians / 2 + edt::Math::DegToRad(state.phase_degrees);

    auto color_fn = app.GetSpawnColorStrategy().GetColorFunction();

    for (size_t i : std::views::iota(size_t{0}, num_directions))
    {
        auto matrix = edt::Math::RotationMatrix2d(
            phase_radians - (sector_radians * static_cast<float>(i)) / static_cast<float>(num_directions));
        auto v = edt::Math::TransformVector(matrix, Vec2f::AxisY());

        VerletObject obj{
            .old_position = config.position + config.radius * v,
            .position =
                config.position + (config.radius + config.speed_factor * constants::kTimeStepDurationSeconds) * v,
            .color = {},
            .scale = Vec2f{} + constants::kObjectRadius,
        };

        obj.color = color_fn(obj);

        app.AddObject(obj);
    }

    state.phase_degrees = NormalizeDegrees(state.phase_degrees + config.rotation_speed);
}

void RadialEmitter::GUI()
{
    ImGui::PushID(this);
    if (ImGui::CollapsingHeader("Radial"))
    {
        DeleteButton();
        ImGui::SameLine();
        CloneButton();
        EnabledCheckbox();

        bool changed = false;

        changed |= klvk::ImGuiHelper::FiniteDragFloat2("Position", config.position, 1.f, 0.f, 0.f, "%.1f");

        if (klvk::ImGuiHelper::FiniteSliderFloat("Phase", config.phase_degrees, -180.f, 180.f, "%.0f deg"))
        {
            config.phase_degrees = NormalizeDegrees(config.phase_degrees);
            changed = true;
        }

        changed |= klvk::ImGuiHelper::FiniteSliderFloat(
            "Sector",
            config.sector_degrees,
            0.f,
            360.f,
            "%.0f deg",
            ImGuiSliderFlags_AlwaysClamp);

        const float max_radius = constants::kWorldRange.Extent().Max();
        changed |= klvk::ImGuiHelper::FiniteDragFloat(
            "Radius",
            config.radius,
            1.f,
            0.f,
            max_radius,
            "%.1f world units",
            ImGuiSliderFlags_AlwaysClamp);

        changed |= klvk::ImGuiHelper::FiniteDragFloat(
            "Speed",
            config.speed_factor,
            0.5f,
            -kMaxSpeed,
            kMaxSpeed,
            "%.1f world units/s",
            ImGuiSliderFlags_AlwaysClamp);

        if (klvk::ImGuiHelper::FiniteDragFloat("Rotation", config.rotation_speed, 0.1f, -10.f, 10.f, "%.1f deg/tick"))
        {
            config.rotation_speed = NormalizeDegrees(config.rotation_speed);
            changed = true;
        }

        if (changed) state = {.phase_degrees = NormalizeDegrees(config.phase_degrees)};
    }
    ImGui::PopID();
}

void RadialEmitter::ResetRuntimeState()
{
    Emitter::ResetRuntimeState();
    state = {.phase_degrees = NormalizeDegrees(config.phase_degrees)};
}

std::unique_ptr<Emitter> RadialEmitter::Clone() const
{
    return std::make_unique<RadialEmitter>(*this);
}

}  // namespace verlet
