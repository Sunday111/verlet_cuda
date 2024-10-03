#include "emitter.hpp"

#include <imgui.h>

namespace verlet
{
void Emitter::DeleteButton()
{
    if (!pending_kill && ImGui::Button("Delete"))
    {
        pending_kill = true;
    }
}
void Emitter::CloneButton()
{
    if (!clone_requested && ImGui::Button("Clone"))
    {
        clone_requested = true;
    }
}

void Emitter::EnabledCheckbox()
{
    ImGui::Checkbox("Enabled", &enabled);
}

void Emitter::ResetRuntimeState()
{
    pending_kill = false;
    clone_requested = false;
    enabled = false;
}

}  // namespace verlet
