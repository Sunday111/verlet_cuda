#include "radial_emitter.hpp"

#include <imgui.h>

#include <algorithm>

#include "EverydayTools/Math/Math.hpp"
#include "coloring/spawn_color/spawn_color_strategy.hpp"
#include "constants.hpp"
#include "cuda_verlet_app.hpp"
#include "klgl/ui/simple_type_widget.hpp"

namespace verlet
{

RadialEmitter::RadialEmitter(const RadialEmitterConfig& in_config) : config(in_config)
{
    state = {.phase_degrees = in_config.phase_degrees};
}

void RadialEmitter::Tick(VerletCudaApp& app)
{
    if (!enabled) return;
    if (app.GetObjectsCount() >= app.GetMaxObjectsCount()) return;

    const float sector_radians = edt::Math::DegToRad(std::clamp(config.sector_degrees, 0.f, 360.f));
    const size_t num_directions = static_cast<size_t>(
        sector_radians * (config.radius + constants::kObjectRadius) / (2 * constants::kObjectRadius));
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

    state.phase_degrees += config.rotation_speed;
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

        bool c = false;
        c |= klgl::SimpleTypeWidget("location", config.position);
        c |= klgl::SimpleTypeWidget("phase degrees", config.phase_degrees);
        c |= klgl::SimpleTypeWidget("sector degrees", config.sector_degrees);
        c |= klgl::SimpleTypeWidget("radius", config.radius);
        c |= klgl::SimpleTypeWidget("speed factor", config.speed_factor);
        c |= klgl::SimpleTypeWidget("rotation speed", config.rotation_speed);

        if (c)
        {
            ResetRuntimeState();
        }
    }
    ImGui::PopID();
}

void RadialEmitter::ResetRuntimeState()
{
    Emitter::ResetRuntimeState();
    state = {.phase_degrees = config.phase_degrees};
}

std::unique_ptr<Emitter> RadialEmitter::Clone() const
{
    return std::make_unique<RadialEmitter>(*this);
}

}  // namespace verlet
