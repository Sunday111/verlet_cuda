#pragma once

#include "CppReflection/GetTypeInfo.hpp"
#include "coloring/object_color_function.hpp"

namespace verlet
{
class VerletCudaApp;
class SpawnColorStrategy
{
public:
    explicit SpawnColorStrategy(const VerletCudaApp& app) : app_{&app} {}
    virtual ~SpawnColorStrategy() = default;
    [[nodiscard]] virtual ObjectColorFunction GetColorFunction() = 0;
    [[nodiscard]] virtual const cppreflection::Type& GetType() const = 0;
    virtual void DrawGUI() {}

    [[nodiscard]] const VerletCudaApp& GetApp() const { return *app_; }

private:
    const VerletCudaApp* app_ = nullptr;
};
}  // namespace verlet
