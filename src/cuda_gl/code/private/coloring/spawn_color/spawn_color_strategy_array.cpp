#include "spawn_color_strategy_array.hpp"

namespace verlet
{
[[nodiscard]] ObjectColorFunction SpawnColorStrategyArray ::GetColorFunction()
{
    return [this]([[maybe_unused]] const VerletObject& object)
    {
        edt::Vec4<uint8_t> c{colors[index].x(), colors[index].y(), colors[index].z(), 255};

        ++index;
        index = index % colors.size();

        return c.Cast<float>() / 255;
    };
}

void SpawnColorStrategyArray::DrawGUI() {}
}  // namespace verlet
