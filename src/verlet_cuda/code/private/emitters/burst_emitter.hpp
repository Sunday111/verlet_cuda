#pragma once

#include "emitter.hpp"

namespace verlet
{
class BurstEmitter : public Emitter
{
public:
    void Tick(VerletCudaApp& app) override;
    void GUI() override;
    void ResetRuntimeState() override;
    [[nodiscard]] std::unique_ptr<Emitter> Clone() const override;
    [[nodiscard]] constexpr EmitterType GetType() const override { return EmitterType::Burst; }

    bool shuffle_storage = true;
    bool packing = true;

private:
    bool emitted_ = false;
    bool does_not_fit_ = false;
};
}  // namespace verlet
