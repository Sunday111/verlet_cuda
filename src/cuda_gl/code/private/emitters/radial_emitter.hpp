#pragma once

#include "EverydayTools/Math/Matrix.hpp"
#include "emitter.hpp"

namespace verlet
{

class RadialEmitterConfig
{
public:
    edt::Vec2f position = {0, 0};
    float radius = 10.f;
    float phase_degrees = 0.f;
    float sector_degrees = 90.f;
    float speed_factor = 10.f;

    // Degrees per tick
    float rotation_speed = 0.f;
};

class RadialEmitterState
{
public:
    float phase_degrees = 0.f;
};

class RadialEmitter : public Emitter
{
public:
    RadialEmitter() = default;
    explicit RadialEmitter(const RadialEmitterConfig& in_config);

    void Tick(VerletCudaApp& app) override;
    void GUI() override;
    std::unique_ptr<Emitter> Clone() const override;
    constexpr EmitterType GetType() const override { return EmitterType::Radial; }
    void ResetRuntimeState() override;

    RadialEmitterConfig config{};
    RadialEmitterState state{};
};
}  // namespace verlet
