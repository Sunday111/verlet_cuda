#pragma once

#include "spawn_color_strategy.hpp"

namespace verlet
{
class SpawnColorStrategyArray : public SpawnColorStrategy
{
public:
    using SpawnColorStrategy::SpawnColorStrategy;
    [[nodiscard]] ObjectColorFunction GetColorFunction() override;
    [[nodiscard]] const cppreflection::Type& GetType() const override
    {
        return *cppreflection::GetTypeInfo<SpawnColorStrategyArray>();
    }
    void DrawGUI() override;

    std::vector<edt::Vec3u8> colors;
    size_t index = 0;
};
}  // namespace verlet

namespace cppreflection
{

template <>
struct TypeReflectionProvider<verlet::SpawnColorStrategyArray>
{
    [[nodiscard]] inline constexpr static auto ReflectType()
    {
        return cppreflection::StaticClassTypeInfo<verlet::SpawnColorStrategyArray>(
            "verlet::SpawnColorStrategyArray ",
            edt::GUID::Create("67A7996F-890A-4070-9980-F3112D8BEF16"));
    }
};

}  // namespace cppreflection
