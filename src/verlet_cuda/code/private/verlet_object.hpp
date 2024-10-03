#pragma once

#include "EverydayTools/Math/Matrix.hpp"

namespace verlet
{
class VerletObject
{
public:
    edt::Vec2f old_position;
    edt::Vec2f position;
    edt::Vec4f color;
    edt::Vec2f scale;
};
}  // namespace verlet
