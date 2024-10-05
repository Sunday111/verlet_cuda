#pragma once

#include "EverydayTools/Math/Matrix.hpp"

namespace verlet
{
inline constexpr uint32_t kInvalidObjectIndex = std::numeric_limits<uint32_t>::max();

class VerletObject
{
public:
    edt::Vec2f old_position;
    edt::Vec2f position;
    edt::Vec4f color;
    edt::Vec2f scale;
    uint32_t next_object_in_cell = kInvalidObjectIndex;
};
}  // namespace verlet
