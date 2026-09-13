#pragma once

#include "edt/math/matrix.hpp"

namespace verlet
{
inline constexpr uint32_t kInvalidObjectIndex = std::numeric_limits<uint32_t>::max();

class VerletObject
{
public:
    edt::Vec2f position;
    uint32_t next_object_in_cell = kInvalidObjectIndex;
};

class VerletAppearance
{
public:
    edt::Vec4f color;
    edt::Vec2f scale;
};

static_assert(sizeof(VerletObject) == 12);
static_assert(sizeof(VerletAppearance) == 24);
}  // namespace verlet
