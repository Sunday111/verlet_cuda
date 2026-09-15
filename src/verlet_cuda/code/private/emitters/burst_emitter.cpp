#include "burst_emitter.hpp"

#include <imgui.h>

#include "burst_layout.hpp"
#include "coloring/spawn_color/spawn_color_strategy.hpp"
#include "verlet_cuda_app.hpp"

namespace verlet
{
void BurstEmitter::Tick(VerletCudaApp& app)
{
    if (!enabled || emitted_) return;
    const size_t count = app.GetRemainingObjectCapacity();
    if (count == 0) return;
    const auto positions = BurstLayout::Generate(count, packing, shuffle_storage);
    does_not_fit_ = positions.size() != count;
    if (does_not_fit_)
    {
        enabled = false;
        return;
    }
    auto color = app.GetSpawnColorStrategy().GetColorFunction();
    for (const auto& position : positions)
    {
        const VerletObject object{.position = position};
        app.AddObject(object, position, {.color = color(object), .scale = Vec2f{} + constants::kObjectRadius});
    }
    emitted_ = true;
}

void BurstEmitter::GUI()
{
    ImGui::PushID(this);
    if (ImGui::CollapsingHeader("Burst"))
    {
        DeleteButton();
        ImGui::SameLine();
        CloneButton();
        EnabledCheckbox();
        ImGui::Checkbox("Shuffle storage", &shuffle_storage);
        ImGui::Checkbox("Packing", &packing);
        ImGui::TextUnformatted("Spawns the entire remaining particle budget once, at rest.");
        if (does_not_fit_)
            ImGui::TextUnformatted("Packed burst does not fit in this world. No particles were spawned.");
        if (ImGui::Button("Rearm"))
        {
            emitted_ = false;
            does_not_fit_ = false;
        }
    }
    ImGui::PopID();
}

void BurstEmitter::ResetRuntimeState()
{
    Emitter::ResetRuntimeState();
    emitted_ = false;
    does_not_fit_ = false;
}

std::unique_ptr<Emitter> BurstEmitter::Clone() const
{
    auto clone = std::make_unique<BurstEmitter>(*this);
    clone->ResetRuntimeState();
    return clone;
}
}  // namespace verlet
