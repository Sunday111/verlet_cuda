#include "spawn_color_strategy_rainbow.hpp"

#include "cuda_verlet_app.hpp"
#include "imgui.h"

namespace verlet
{
[[nodiscard]] ObjectColorFunction SpawnColorStrategyRainbow ::GetColorFunction()
{
    return [t = phase_ + frequency_ * GetApp().GetTimeSeconds()]([[maybe_unused]] const VerletObject& object)
    {
        auto rgb = edt::Math::GetRainbowColors(t);
        Vec4u8 c;
        c.x() = rgb.x();
        c.y() = rgb.y();
        c.z() = rgb.z();
        c.w() = 255;
        return c.Cast<float>() / 255;
    };
}

void SpawnColorStrategyRainbow::DrawGUI()
{
    ImGui::SliderFloat("Phase", &phase_, -10.f, 10.f);
    ImGui::SliderFloat("Frequency", &frequency_, 0.f, 2.f);
}
}  // namespace verlet
