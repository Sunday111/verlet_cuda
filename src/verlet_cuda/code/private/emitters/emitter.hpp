#pragma once

#include <memory>

#include "emitter_type.hpp"

namespace verlet
{

class VerletCudaApp;

class Emitter
{
public:
    virtual void Tick(VerletCudaApp& app) = 0;
    virtual void GUI() = 0;
    virtual constexpr EmitterType GetType() const = 0;
    virtual std::unique_ptr<Emitter> Clone() const = 0;
    virtual void ResetRuntimeState();
    virtual ~Emitter() = default;

    void DeleteButton();
    void CloneButton();
    void EnabledCheckbox();

    bool pending_kill = false;
    bool clone_requested = false;
    bool enabled = false;
};

}  // namespace verlet
