#pragma once

#include "spawn_color_strategy.hpp"

namespace verlet
{
class SpawnColorStrategyRainbow : public SpawnColorStrategy
{
public:
    using SpawnColorStrategy::SpawnColorStrategy;
    [[nodiscard]] ObjectColorFunction GetColorFunction() override;
    [[nodiscard]] const cppreflection::Type& GetType() const override
    {
        return *cppreflection::GetTypeInfo<SpawnColorStrategyRainbow>();
    }
    void DrawGUI() override;

private:
    float phase_ = 0.f;
    float frequency_ = 1.f;
};
}  // namespace verlet

namespace cppreflection
{

template <>
struct TypeReflectionProvider<verlet::SpawnColorStrategyRainbow>
{
    [[nodiscard]] inline constexpr static auto ReflectType()
    {
        return cppreflection::StaticClassTypeInfo<verlet::SpawnColorStrategyRainbow>(
            "verlet::SpawnColorStrategyRainbow ",
            edt::GUID::Create("B1EFD067-A712-4EB3-91E3-70003B18B5F4"));
    }
};

}  // namespace cppreflection
